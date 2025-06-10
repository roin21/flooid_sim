class_name GPUSort
extends RefCounted

# --- Compute Helpers  ---
# using a plugin because I am a numpty and shader dispatching was breaking my brain
var sort_compute: ComputeHelper
var offsets_compute: ComputeHelper
var reset_compute: ComputeHelper

# --- Class members ---
var num_entries: int
var padded_size: int

const WORKGROUP_SIZE = 256 # must be a power of two (256 might be too high for low-end hardware, not sure tbh)

func setup(indices_uniform: StorageBufferUniform, offsets_uniform: StorageBufferUniform, p_num_entries: int):
	"""Initializes the GPU sorting system with the provided uniforms and number of entries."""
	num_entries = p_num_entries
	padded_size = next_power_of_two(num_entries) # padding the size to the next power of two for bitonic sort

	# bitonic sorting shader
	sort_compute = ComputeHelper.create("res://shaders/BitonicSort.glsl")
	sort_compute.add_uniform(indices_uniform)

	# shader for calculating offsets
	offsets_compute = ComputeHelper.create("res://shaders/CalculateOffsets.glsl")
	offsets_compute.add_uniform_array([indices_uniform, offsets_uniform])
	
	# shader for resetting offsets
	# this could be done on the CPU, but this is way faster. Might need to switch if this breaks things
	reset_compute = ComputeHelper.create("res://shaders/ResetOffsets.glsl")
	reset_compute.add_uniform(offsets_uniform)

func sort_entries():
	var stages = int(log(padded_size) / log(2)) # number of stages in the bitonic sort

	for stage in range(stages):
		for step in range(stage + 1):
			var push_constants = PackedByteArray()
			push_constants.append_array(_encode_uint32(padded_size)) # total number of entries to sort
			push_constants.append_array(_encode_uint32(1 << (stage + 1))) # k = 2^(stage + 1)
			push_constants.append_array(_encode_uint32(1 << (stage - step))) # j = 2^(stage - step)

			var num_threads = padded_size / 2 
			var thread_groups = Vector3i((num_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1)
			sort_compute.run(thread_groups, push_constants)
			ComputeHelper.sync() #! this is probably killing performance. haven't tested without it yet
	
	#? old logic that might be useful later
	# var num_to_sort = next_power_of_two(num_entries)
	# var k = 2
	# while k <= num_to_sort:
	# 	var j = k / 2
	# 	while j > 0:
	# 		var push_constants = PackedByteArray()
	# 		push_constants.append_array(_encode_uint32(num_to_sort))
	# 		push_constants.append_array(_encode_uint32(k))
	# 		push_constants.append_array(_encode_uint32(j))
	# 		# push_constants.append_array(_encode_uint32(0)) # Padding

	# 		var num_threads = num_to_sort / 2
	# 		var thread_groups = Vector3i((num_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1)

	# 		sort_compute.run(thread_groups, push_constants)
	# 		ComputeHelper.sync()

	# 		j /= 2
	# 	k *= 2

# fills the offsets buffer with a large value before calculation.
func reset_offsets():
	var push_constants = PackedByteArray()
	push_constants.append_array(_encode_uint32(num_entries))

	var thread_groups = Vector3i((num_entries + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1)
	reset_compute.run(thread_groups, push_constants)
	ComputeHelper.sync() #! again, this is probably killing performance

# Calculates the start index for each key in the sorted buffer.
func calculate_offsets():
	var push_constants = PackedByteArray()
	push_constants.append_array(_encode_uint32(num_entries))

	var thread_groups = Vector3i((num_entries + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1)
	offsets_compute.run(thread_groups, push_constants)
	ComputeHelper.sync() #! performance killer

# --- Helper Functions ---
func _encode_uint32(value: int) -> PackedByteArray:
	var bytes = PackedByteArray()
	bytes.resize(4)
	bytes.encode_u32(0, value)
	return bytes

func next_power_of_two(n: int) -> int:
	if n == 0:
		return 1
	var power = 1
	while power < n:
		power *= 2
	return power