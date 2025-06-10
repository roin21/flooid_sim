class_name ParticleDisplay2D
extends Node2D

@export var particle_scale: float = 8.0
@export var velocity_display_max: float = 5.0
@export var color_ramp: Gradient
@export var update_frequency: int = 1  

var simulation: Simulation2D
var multimesh_instance: MultiMeshInstance2D
var multimesh: MultiMesh
var frame_counter: int = 0

# Debug visualization
@export var debug_draw: bool = true
@export var debug_particle_count: int = 10  # Number of particles to debug draw

func init(sim: Simulation2D):
	simulation = sim
	
	# multimesh for efficient rendering
	multimesh = MultiMesh.new()
	multimesh.use_colors = true 
	multimesh.mesh = QuadMesh.new()
	multimesh.mesh.size = Vector2.ONE * particle_scale
	multimesh.transform_format = MultiMesh.TRANSFORM_2D
	multimesh.instance_count = simulation.num_particles
	
	# next we instance the multimesh
	multimesh_instance = MultiMeshInstance2D.new()
	multimesh_instance.multimesh = multimesh
	multimesh_instance.material = preload("res://materials/particle.tres") # material has a shader for pretty colors based on velocity
	add_child(multimesh_instance)
	
	# initialize color gradient (kinda ugly rn)
	if not color_ramp:
		color_ramp = Gradient.new()
		color_ramp.add_point(0.0, Color.BLUE)
		color_ramp.add_point(0.5, Color.GREEN)
		color_ramp.add_point(1.0, Color.RED)
	
	print("ParticleDisplay2D initialized with ", simulation.num_particles, " particles")

func _process(_delta):
	if not simulation:
		return
	
	frame_counter += 1
	
	# Update display based on frequency setting
	if frame_counter % update_frequency == 0:
		_update_display()
	
	# Debug draw
	if debug_draw:
		queue_redraw()

func _update_display():
	# get buffer RIDs from simulation
	var position_buffer = simulation.get_position_buffer()
	var velocity_buffer = simulation.get_velocity_buffer()
	
	if not position_buffer.is_valid() or not velocity_buffer.is_valid():
		push_error("Invalid buffers in ParticleDisplay2D")
		return
	
	# read data from GPU (i feel like there's a faster way to do this)
	var position_data = ComputeHelper.rd.buffer_get_data(position_buffer)
	var velocity_data = ComputeHelper.rd.buffer_get_data(velocity_buffer)
	
	# verify data size
	#! this might need to be updated with the power of two padding
	var expected_size = simulation.num_particles * 8  # 2 floats * 4 bytes each
	if position_data.size() != expected_size:
		push_error("Position buffer size mismatch: expected ", expected_size, " got ", position_data.size())
		return
	
	for i in range(simulation.num_particles):
		var pos = Vector2(
			position_data.decode_float(i * 8),
			position_data.decode_float(i * 8 + 4)
		)
		var vel = Vector2(
			velocity_data.decode_float(i * 8),
			velocity_data.decode_float(i * 8 + 4)
		)
		
		# check for NaN or very large values
		if is_nan(pos.x) or is_nan(pos.y) or is_inf(pos.x) or is_inf(pos.y):
			push_error("NaN/Inf position detected at particle ", i, ": ", pos)
			pos = Vector2.ZERO  # Reset to origin
		
		# Set transform
		var p_transform = Transform2D()
		p_transform.origin = pos * 10.0  # Scale up for visibility
		multimesh.set_instance_transform_2d(i, p_transform)
		
		# Set color based on velocity
		var speed = vel.length()
		var color_t = clamp(speed / velocity_display_max, 0.0, 1.0)
		var color = color_ramp.sample(color_t)
		multimesh.set_instance_color(i, color)

func _draw():
	if not debug_draw or not simulation:
		return
	
	# simulation bounds
	#? this needs a rework to make sure the coords are aligned with everything else in the sim (particularly the spawning grid)
	var bounds = simulation.bounds_size * 10.0  # Scale up
	draw_rect(Rect2(-bounds * 0.5, bounds), Color.GREEN, false, 2.0)
	
	var position_buffer = simulation.get_position_buffer()
	if position_buffer.is_valid():
		var position_data = ComputeHelper.rd.buffer_get_data(position_buffer)
		
		for i in range(min(debug_particle_count, simulation.num_particles)):
			var pos = Vector2(
				position_data.decode_float(i * 8),
				position_data.decode_float(i * 8 + 4)
			) * 10.0  # Scale up
			
			# Draw particle
			draw_circle(pos, particle_scale * 10.0, Color.YELLOW)
			
