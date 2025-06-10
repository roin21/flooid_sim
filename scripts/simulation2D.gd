class_name Simulation2D
extends Node

#? this script needs some refactoring, it a mess

signal simulation_step_completed

# simulation settings
@export_group("Simulation Settings")
@export var time_scale: float = 1.0
@export var fixed_timestep: bool = false
@export var iterations_per_frame: int = 1
@export var gravity: float = 6.0 
@export_range(0.0, 1.0) var collision_damping: float = 0.6
@export var smoothing_radius: float = 1.9
@export var target_density: float = 1.0
@export var pressure_multiplier: float = 10.0
@export var near_pressure_multiplier: float = 20.0
@export var viscosity_strength: float = 0.5
@export var bounds_size: Vector2 = Vector2(40, 30)
@export var obstacle_size: Vector2 = Vector2(0, 0)
@export var obstacle_center: Vector2 = Vector2.ZERO

@export_group("Interaction Settings")
@export var interaction_radius: float = 3.0
@export var interaction_strength: float = 10.0

@export_group("Debug Settings")
@export var debug_mode: bool = false
@export var pause_after_frames: int = 1
@export var print_particle_data: bool = true
@export var skip_kernels: Array[bool] = [false, false, false, false, false, false]

@export_group("References")
@export var particle_spawner: ParticleSpawner
@export var particle_display: ParticleDisplay2D

# compute shader
var compute_shader: ComputeHelper
var kernel_uniform_sets: Array = []

# buffers
var position_uniform: StorageBufferUniform
var predicted_position_uniform: StorageBufferUniform
var velocity_uniform: StorageBufferUniform
var density_uniform: StorageBufferUniform
var spatial_indices_uniform: StorageBufferUniform
var spatial_offsets_uniform: StorageBufferUniform

# GPU sort
var gpu_sort: GPUSort

# kernel indices
const KERNEL_EXTERNAL_FORCES = 0
const KERNEL_SPATIAL_HASH = 1
const KERNEL_CALCULATE_DENSITIES = 2
const KERNEL_PRESSURE_FORCE = 3
const KERNEL_VISCOSITY = 4
const KERNEL_UPDATE_POSITIONS = 5

const KERNEL_NAMES = [
	"External Forces",
	"Spatial Hash",
	"Calculate Densities",
	"Pressure Force",
	"Viscosity",
	"Update Positions"
]

# state
var is_paused: bool = false
var num_particles: int = 0
var spawn_data: Dictionary
var current_delta_time: float = 0.0
var frame_count: int = 0
var prev_physical_keys: Dictionary = {}

func _ready() -> void:
	print("=== Simulation2D Ready ===")
	spawn_data = particle_spawner.get_spawn_data()
	num_particles = spawn_data.positions.size()
	print("Number of particles: ", num_particles)
	print("Spawn bounds: ", particle_spawner.spawn_size)
	print("Smoothing radius: ", smoothing_radius)
	
	_create_buffers()
	_setup_compute_shader()

	_verify_simulation_parameters()
	_test_kernel_functions()

	gpu_sort = GPUSort.new()
	gpu_sort.setup(spatial_indices_uniform, spatial_offsets_uniform, num_particles)

	particle_display.init(self)

	if debug_mode:
		is_paused = true
		print("Simulation paused for debugging")
	
	set_physics_process(fixed_timestep)
	set_process(not fixed_timestep)

	if debug_mode and print_particle_data:
		_print_debug_info("Initial State")

func _create_buffers():
	var vec2_size = 8 # size of Vector2 in bytes
	var spatial_index_size = 16 # ditched the uvec3, new spatial index is 4 u32s (last one is padding for alignment)
	var uint_size = 4

	var position_data = PackedByteArray()
	position_data.resize(num_particles * vec2_size)
	for i in range(num_particles):
		var pos = spawn_data.positions[i]
		position_data.encode_float(i * vec2_size, pos.x)
		position_data.encode_float(i * vec2_size + 4, pos.y)
	position_uniform = StorageBufferUniform.create(position_data)
	predicted_position_uniform = StorageBufferUniform.create(position_data)

	var velocity_data = PackedByteArray()
	velocity_data.resize(num_particles * vec2_size)
	for i in range(num_particles):
		var vel = spawn_data.velocities[i]
		velocity_data.encode_float(i * vec2_size, vel.x)
		velocity_data.encode_float(i * vec2_size + 4, vel.y)
	velocity_uniform = StorageBufferUniform.create(velocity_data)

	var density_data = PackedByteArray()
	density_data.resize(num_particles * vec2_size)
	density_uniform = StorageBufferUniform.create(density_data)


	var padded_size = next_power_of_two(num_particles)
	var spatial_indices_data = PackedByteArray()
	spatial_indices_data.resize(padded_size * spatial_index_size)
	for i in range(padded_size):
		var byte_offset = i * spatial_index_size
		spatial_indices_data.encode_u32(byte_offset, i) # originalIndex = i  
		spatial_indices_data.encode_u32(byte_offset + 4, 0)  # Placeholder for hash
		spatial_indices_data.encode_u32(byte_offset + 8, 0)  # Placeholder for key
		spatial_indices_data.encode_u32(byte_offset + 12, 0) # Padding for alignment
	spatial_indices_uniform = StorageBufferUniform.create(spatial_indices_data)

	var spatial_offsets_data = PackedByteArray()
	spatial_offsets_data.resize(num_particles * uint_size)
	for i in range(num_particles):
		spatial_offsets_data.encode_u32(i * uint_size, num_particles)
	spatial_offsets_uniform = StorageBufferUniform.create(spatial_offsets_data)

func next_power_of_two(n: int) -> int:
	if n == 0:
		return 1
	var power = 1
	while power < n:
		power *= 2
	return power

func _setup_compute_shader():
	var shader_file = "res://shaders/FluidSim2d.glsl"
	if not FileAccess.file_exists(shader_file):
		push_error("Compute shader file not found: ", shader_file)
		return
	
	compute_shader = ComputeHelper.create(shader_file)

	compute_shader.add_uniform_array([
		position_uniform,
		predicted_position_uniform,
		velocity_uniform,
		density_uniform,
		spatial_indices_uniform,
		spatial_offsets_uniform
	])

func _physics_process(delta: float) -> void:
	if fixed_timestep:
		run_simulation_frame(delta)
	
		
func _process(delta: float) -> void:
	if not fixed_timestep and Engine.get_physics_frames() > 10:
		run_simulation_frame(delta)
	
	handle_input()

func run_simulation_frame(frame_time: float):
	if not is_paused:
		var time_step = frame_time / iterations_per_frame * time_scale
		current_delta_time = time_step

		if debug_mode:
			print("\n=== Frame ", frame_count, " (delta: ", time_step, ") ===")

		for i in range(iterations_per_frame):
			run_simulation_step()
			simulation_step_completed.emit()
		
		frame_count += 1

		if debug_mode and pause_after_frames > 0 and frame_count >= pause_after_frames:
			is_paused = true
			print("Simulation paused after ", frame_count, " frames")
			_print_debug_info("After Frame " + str(frame_count))

func run_simulation_step():
	var thread_groups = Vector3i((num_particles + 63) / 64, 1, 1)

	for kernel_index in range(6):
		if skip_kernels[kernel_index]:
			if debug_mode:
				print("Skipping kernel: ", KERNEL_NAMES[kernel_index])
			continue
		
		if debug_mode:
			print(" Running kernel: ", KERNEL_NAMES[kernel_index])

		var push_constant_data = _get_push_constants_data(kernel_index)
		compute_shader.run(thread_groups, push_constant_data)

		if debug_mode:
			ComputeHelper.sync()
			_check_for_nan_inf(kernel_index)

		if kernel_index == KERNEL_SPATIAL_HASH:
			gpu_sort.reset_offsets()
			ComputeHelper.sync() #! this is called in reset_offsets already, need to sort out when its actually needed

			gpu_sort.sort_entries()
			gpu_sort.calculate_offsets()

			if debug_mode:
				ComputeHelper.sync()
				_debug_spatial_hash()
	
	if not debug_mode:
		ComputeHelper.sync()

func _check_for_nan_inf(kernel_index: int):
	if not debug_mode:
		return
		
	# Check positions
	var pos_data = position_uniform.get_data()
	for i in range(min(5, num_particles)):  # Check first 5 particles
		var x = pos_data.decode_float(i * 8)
		var y = pos_data.decode_float(i * 8 + 4)
		if is_nan(x) or is_nan(y) or is_inf(x) or is_inf(y):
			push_error("NaN/Inf detected in position after " + KERNEL_NAMES[kernel_index] + " at particle " + str(i) + ": " + str(Vector2(x, y)))
			is_paused = true
			break
	
	# Check velocities
	var vel_data = velocity_uniform.get_data()
	for i in range(min(5, num_particles)):
		var vx = vel_data.decode_float(i * 8)
		var vy = vel_data.decode_float(i * 8 + 4)
		if is_nan(vx) or is_nan(vy) or is_inf(vx) or is_inf(vy):
			push_error("NaN/Inf detected in velocity after " + KERNEL_NAMES[kernel_index] + " at particle " + str(i) + ": " + str(Vector2(vx, vy)))
			is_paused = true
			break

func _print_debug_info(label: String):
	print("\n=== Debug Info: ", label, " ===")
	
	# Print first few particle positions
	var pos_data = position_uniform.get_data()
	var vel_data = velocity_uniform.get_data()
	var density_data = density_uniform.get_data()
	
	print("First 5 particles:")
	for i in range(min(5, num_particles)):
		var pos = Vector2(
			pos_data.decode_float(i * 8),
			pos_data.decode_float(i * 8 + 4)
		)
		var vel = Vector2(
			vel_data.decode_float(i * 8),
			vel_data.decode_float(i * 8 + 4)
		)
		var density = Vector2(
			density_data.decode_float(i * 8),
			density_data.decode_float(i * 8 + 4)
		)
		print("  Particle ", i, ": pos=", pos, ", vel=", vel, ", density=", density.x, ", nearDensity=", density.y)
	
	# Check bounds
	var out_of_bounds = 0
	for i in range(num_particles):
		var x = pos_data.decode_float(i * 8)
		var y = pos_data.decode_float(i * 8 + 4)
		if abs(x) > bounds_size.x * 0.5 or abs(y) > bounds_size.y * 0.5:
			out_of_bounds += 1
	
	if out_of_bounds > 0:
		print("WARNING: ", out_of_bounds, " particles out of bounds!")

func _debug_spatial_hash(): #? pretty sure this is not fully correct, especially the sorting
	print("\n=== Spatial Hash Debug ===")
	
	var indices_data = spatial_indices_uniform.get_data()
	var offsets_data = spatial_offsets_uniform.get_data()
	
	var spatial_index_size = 16  # Now 16 bytes due to padding
	
	# check first few entries with 16-byte alignment
	print("\nFirst 10 sorted entries:")
	for i in range(min(10, num_particles)):
		var byte_offset = i * spatial_index_size
		var original_index = indices_data.decode_u32(byte_offset)
		var hash_value = indices_data.decode_u32(byte_offset + 4)
		var key = indices_data.decode_u32(byte_offset + 8)
		
		# sanity check
		if original_index >= num_particles:
			print("  [%d] ERROR: original_index=%d (>= %d), hash=%d, key=%d" % 
				[i, original_index, num_particles, hash_value, key])
		else:
			print("  [%d] original_index=%d, hash=%d, key=%d" % [i, original_index, hash_value, key])
	
	# Check offsets
	print("\nOffset values (non-default):")
	var found_offsets = []
	for i in range(min(num_particles, offsets_data.size() / 4)):
		var offset = offsets_data.decode_u32(i * 4)
		if offset < num_particles:
			found_offsets.append("offsets[%d] = %d" % [i, offset])
			if found_offsets.size() >= 20:
				break
	print("  ", found_offsets)
	
	# verify sorting by key
	var prev_key = -1
	var sort_errors = 0
	for i in range(min(num_particles, indices_data.size() / spatial_index_size)):
		var key = indices_data.decode_u32(i * spatial_index_size + 8)
		if key < prev_key and prev_key != -1:
			sort_errors += 1
			if sort_errors < 5:
				print("  ERROR: Sort order violated at index %d (key %d < prev %d)" % [i, key, prev_key])
		prev_key = key
	
	if sort_errors > 0:
		print("  Total sort errors: ", sort_errors)
	else:
		print("  Sort order verified!")
	
	# Count particles per cell
	var cells = {}
	var valid_particles = 0
	for i in range(min(num_particles, indices_data.size() / spatial_index_size)):
		var original_index = indices_data.decode_u32(i * spatial_index_size)
		var key = indices_data.decode_u32(i * spatial_index_size + 8)
		
		if original_index < num_particles:
			cells[key] = cells.get(key, 0) + 1
			valid_particles += 1
	
	print("\nValid particles found: %d / %d" % [valid_particles, num_particles])
	print("Cell occupancy (showing first 10 cells with particles):")
	var sorted_keys = cells.keys()
	sorted_keys.sort()
	for i in range(min(10, sorted_keys.size())):
		var key = sorted_keys[i]
		print("  Cell key %d: %d particles" % [key, cells[key]])

func _verify_simulation_parameters(): #? this function was entirely vibecoded ngl
	print("\n=== Simulation Parameters Check ===")
	
	# Calculate particle spacing
	var total_particles = num_particles
	var spawn_area = particle_spawner.spawn_size.x * particle_spawner.spawn_size.y
	var avg_particle_spacing = sqrt(spawn_area / total_particles)
	
	print("Number of particles: ", num_particles)
	print("Spawn size: ", particle_spawner.spawn_size)
	print("Spawn area: ", spawn_area)
	print("Average particle spacing: %.3f" % avg_particle_spacing)
	print("Smoothing radius: ", smoothing_radius)
	print("Ratio (smoothing_radius / spacing): %.2f" % (smoothing_radius / avg_particle_spacing))
	
	if smoothing_radius < avg_particle_spacing * 1.5:
		print("WARNING: Smoothing radius might be too small relative to particle spacing!")
		print("  Recommended smoothing radius: %.2f to %.2f" % [avg_particle_spacing * 2, avg_particle_spacing * 3])
	
	print("Bounds size: ", bounds_size)
	print("Target density: ", target_density)
	
	# Check if particles fit in bounds
	if particle_spawner.spawn_size.x > bounds_size.x * 0.8 or particle_spawner.spawn_size.y > bounds_size.y * 0.8:
		print("WARNING: Spawn area is very close to simulation bounds!")
	
	# Estimate expected density
	var particles_per_radius = PI * smoothing_radius * smoothing_radius / (avg_particle_spacing * avg_particle_spacing)
	print("Estimated particles within smoothing radius: %.1f" % particles_per_radius)
	
	# Check kernel scaling factors
	var poly6_factor = 4.0 / (PI * pow(smoothing_radius, 8))
	var spiky_pow2_factor = 6.0 / (PI * pow(smoothing_radius, 4))
	print("Poly6 scaling factor: %.6f" % poly6_factor)
	print("Spiky pow2 scaling factor: %.6f" % spiky_pow2_factor)
	
	# Estimate density at particle center with current parameters
	var self_density_contribution = spiky_pow2_factor * smoothing_radius * smoothing_radius
	print("Self density contribution: %.3f" % self_density_contribution)
	print("  (Target density is %.3f)" % target_density)

func _test_kernel_functions(): #? another vibecoded monstrosity
	print("\n=== Testing Kernel Functions ===")
	
	var radius = smoothing_radius
	var spiky_pow2_factor = 6.0 / (PI * pow(radius, 4))
	var spiky_pow3_factor = 10.0 / (PI * pow(radius, 5))
	
	print("Testing SpikyKernelPow2 with radius=%.2f, factor=%.6f" % [radius, spiky_pow2_factor])
	
	# Test at various distances
	var test_distances = [0.0, 0.5, 1.0, 1.5, 1.9, 2.0, 2.1]
	for dst in test_distances:
		var kernel_value = 0.0
		if dst < radius:
			var v = radius - dst
			kernel_value = v * v * spiky_pow2_factor
		print("  Distance %.2f: kernel value = %.6f" % [dst, kernel_value])
	
	print("\nExpected density calculation for uniform grid:")
	var particle_spacing = sqrt(particle_spawner.spawn_size.x * particle_spawner.spawn_size.y / num_particles)
	print("  Particle spacing: %.3f" % particle_spacing)
	
	# Count neighbors within radius
	var expected_neighbors = 0
	var total_density = 0.0
	
	# Check in a grid pattern
	var check_range = int(ceil(radius / particle_spacing))
	for dx in range(-check_range, check_range + 1):
		for dy in range(-check_range, check_range + 1):
			var offset = Vector2(dx * particle_spacing, dy * particle_spacing)
			var dist = offset.length()
			if dist < radius:
				expected_neighbors += 1
				if dist < radius:
					var v = radius - dist
					total_density += v * v * spiky_pow2_factor
	
	print("  Expected neighbors within radius: ", expected_neighbors)
	print("  Expected total density: %.3f" % total_density)
	print("  Target density setting: %.3f" % target_density)
	
	if abs(total_density - target_density) > target_density * 0.5:
		print("  WARNING: Target density might be inappropriate for current particle spacing!")

func _get_push_constants_data(kernel_index: int) -> PackedByteArray:
	var params = PackedByteArray()

	# must be quantized to 16 bytes (4 floats)
	# first 16 bytes
	params.append_array(_encode_float(current_delta_time))
	params.append_array(_encode_float(gravity))
	params.append_array(_encode_float(collision_damping))
	params.append_array(_encode_float(smoothing_radius))

	# second 16 bytes
	params.append_array(_encode_float(target_density))
	params.append_array(_encode_float(pressure_multiplier))
	params.append_array(_encode_float(near_pressure_multiplier))
	params.append_array(_encode_float(viscosity_strength))

	# third 16 bytes
	params.append_array(_encode_float(bounds_size.x))
	params.append_array(_encode_float(bounds_size.y))
	params.append_array(_encode_float(obstacle_size.x))
	params.append_array(_encode_float(obstacle_size.y))

	# fourth 16 bytes
	params.append_array(_encode_float(obstacle_center.x))
	params.append_array(_encode_float(obstacle_center.y))
	params.append_array(_encode_uint32(num_particles))
	params.append_array(_encode_uint32(kernel_index))

	var world_pos = get_viewport().get_camera_2d().get_global_mouse_position()
	var interaction_strength_current = 0.0
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		interaction_strength_current = interaction_strength
	elif Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
		interaction_strength_current = -interaction_strength

	# fifth 16 bytes #! this is also broken
	params.append_array(_encode_float(world_pos.x))
	params.append_array(_encode_float(world_pos.y))
	params.append_array(_encode_float(interaction_strength_current))
	params.append_array(_encode_float(interaction_radius))

	var poly6_factor = 4.0 / (PI * pow(smoothing_radius, 8))
	var spiky_pow3_factor = 10.0 / (PI * pow(smoothing_radius, 5))
	var spiky_pow2_factor = 6.0 / (PI * pow(smoothing_radius, 4))
	var spiky_pow3_deriv_factor = 30.0 / (pow(smoothing_radius, 5) * PI)
	# sixth 16 bytes
	params.append_array(_encode_float(poly6_factor))
	params.append_array(_encode_float(spiky_pow3_factor))
	params.append_array(_encode_float(spiky_pow2_factor))
	params.append_array(_encode_float(spiky_pow3_deriv_factor))

	# seventh 16 bytes
	var spiky_pow2_deriv_factor = 12.0 / (pow(smoothing_radius, 4) * PI)
	params.append_array(_encode_float(spiky_pow2_deriv_factor))
	params.append_array(_encode_float(0.0)) # padding
	params.append_array(_encode_float(0.0)) # "
	params.append_array(_encode_float(0.0)) # "

	return params

func _encode_float(value: float) -> PackedByteArray:
	var bytes = PackedByteArray()
	bytes.resize(4)
	bytes.encode_float(0, value)
	return bytes

func _encode_uint32(value: int) -> PackedByteArray:
	var bytes = PackedByteArray()
	bytes.resize(4)
	bytes.encode_u32(0, value)
	return bytes

func handle_input():
	if Input.is_action_just_pressed("ui_accept"):
		is_paused = not is_paused
	
	if Input.is_key_pressed(KEY_R):
		reset_simulation()
	
	if Input.is_action_just_pressed("ui_right"):
		if is_paused:
			is_paused = false
			pause_after_frames = frame_count + 1
	
	if debug_mode:		
		# Check each debug key using the helper
		check_physical_key_just_pressed(KEY_1, 0, "External Forces")
		check_physical_key_just_pressed(KEY_2, 1, "Spatial Hash")
		check_physical_key_just_pressed(KEY_3, 2, "Calculate Densities")
		check_physical_key_just_pressed(KEY_4, 3, "Pressure Force")
		check_physical_key_just_pressed(KEY_5, 4, "Viscosity")
		check_physical_key_just_pressed(KEY_6, 5, "Update Positions")

# Helper function to check physical key just pressed
func check_physical_key_just_pressed(key, index, message):
	var current = Input.is_physical_key_pressed(key)
	var prev = prev_physical_keys.get(key, false)
	if current and not prev:
		skip_kernels[index] = not skip_kernels[index]
		print(message, ": ", "ON" if not skip_kernels[index] else "OFF")
	prev_physical_keys[key] = current

func reset_simulation():
	is_paused = true
	frame_count = 0
	_set_initial_buffer_data()
	if debug_mode:
		print("Simulation reset to initial state")
		_print_debug_info("After Reset")

func _set_initial_buffer_data():
	var spatial_index_size = 16

	var position_data = PackedByteArray()
	position_data.resize(num_particles * 8)
	for i in range(num_particles):
		var pos = spawn_data.positions[i]
		position_data.encode_float(i * 8, pos.x)
		position_data.encode_float(i * 8 + 4, pos.y)
	position_uniform.update_data(position_data)
	predicted_position_uniform.update_data(position_data)

	var velocity_data = PackedByteArray()
	velocity_data.resize(num_particles * 8)
	for i in range(num_particles):
		var vel = spawn_data.velocities[i]
		velocity_data.encode_float(i * 8, vel.x)
		velocity_data.encode_float(i * 8 + 4, vel.y)
	velocity_uniform.update_data(velocity_data)
	
	var density_data = PackedByteArray()
	density_data.resize(num_particles * 8)
	for i in range(num_particles * 8):
		density_data[i] = 0
	density_uniform.update_data(density_data)

	var padded_size = next_power_of_two(num_particles)
	var spatial_indices_data = PackedByteArray()
	spatial_indices_data.resize(padded_size * spatial_index_size)
	for i in range(padded_size):
		var byte_offset = i * spatial_index_size
		spatial_indices_data.encode_u32(byte_offset, i)       # originalIndex = i
		spatial_indices_data.encode_u32(byte_offset + 4, 0)   # hash = 0
		spatial_indices_data.encode_u32(byte_offset + 8, 0)   # key = 0
		spatial_indices_data.encode_u32(byte_offset + 12, 0)  # padding for alignment
	spatial_indices_uniform.update_data(spatial_indices_data)
	
	# Reset spatial offsets
	var spatial_offsets_data = PackedByteArray()
	spatial_offsets_data.resize(num_particles * 4)
	for i in range(num_particles):
		spatial_offsets_data.encode_u32(i * 4, num_particles)
	spatial_offsets_uniform.update_data(spatial_offsets_data)

# getters for buffers
func get_position_buffer() -> RID:
	return position_uniform.storage_buffer

func get_velocity_buffer() -> RID:
	return velocity_uniform.storage_buffer

func get_density_buffer() -> RID:
	return density_uniform.storage_buffer
