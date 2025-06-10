class_name ParticleSpawner
extends Node2D

@export var particle_count: int = 256
@export var initial_velocity: Vector2 = Vector2.ZERO
@export var spawn_center: Vector2 = Vector2.ZERO
@export var spawn_size: Vector2 = Vector2(18, 18)
@export var jitter_strength: float = 0.0
@export var show_spawn_bounds_gizmos: bool = true

func get_spawn_data() -> Dictionary:
	var positions = []
	var velocities = []
	
	var rng = RandomNumberGenerator.new()
	rng.seed = 42
	
	# Calculate grid dimensions
	var num_x = ceil(sqrt(spawn_size.x / spawn_size.y * particle_count + 
		pow(spawn_size.x - spawn_size.y, 2) / (4 * spawn_size.y * spawn_size.y)) - 
		(spawn_size.x - spawn_size.y) / (2 * spawn_size.y))
	var num_y = ceil(particle_count / float(num_x))
	
	var i = 0
	for y in range(num_y):
		for x in range(num_x):
			if i >= particle_count:
				break
			
			var tx = 0.5 if num_x <= 1 else x / (num_x - 1.0)
			var ty = 0.5 if num_y <= 1 else y / (num_y - 1.0)
			
			# Add jitter
			var angle = rng.randf() * TAU
			var jitter = Vector2(cos(angle), sin(angle)) * jitter_strength * (rng.randf() - 0.5)
			
			var pos = Vector2(
				(tx - 0.5) * spawn_size.x,
				(ty - 0.5) * spawn_size.y
			) + jitter + spawn_center
			
			positions.append(pos)
			velocities.append(initial_velocity)
			i += 1
	
	return {
		"positions": positions,
		"velocities": velocities
	}

func _draw():
	if show_spawn_bounds_gizmos and not Engine.is_editor_hint():
		draw_rect(Rect2(spawn_center - spawn_size * 0.5, spawn_size), Color.YELLOW, false, 2.0)
