#[compute]
#version 450

// GLSL doesn't natively support #includes, and godot doesn't support compute shaders other than raw glsl,
// so we have to define everything in one file

// ========== Kernel Selection ==========
#define KERNEL_EXTERNAL_FORCES 0
#define KERNEL_SPATIAL_HASH 1
#define KERNEL_CALCULATE_DENSITIES 2
#define KERNEL_PRESSURE_FORCE 3
#define KERNEL_VISCOSITY 4
#define KERNEL_UPDATE_POSITIONS 5

// ========== Hashing ==========
const ivec2 offsets2D[9] = ivec2[](
    ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1),
    ivec2(-1, 0), ivec2(0, 0), ivec2(1, 0),
    ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1)
);
const uint hashK1 = 15823;
const uint hashK2 = 9737333;

struct SpatialEntry {
    uint originalIndex; // Original particle index
    uint hash;          // Hash of the spatial cell
    uint key;           // Key for the spatial hash table
    uint _padding;      // Padding for alignment
};

// ========== Push Constants ==========
layout(push_constant) uniform PushConstants {
    float deltaTime;
    float gravity;
    float collisionDamping;
    float smoothingRadius;
    float targetDensity;
    float pressureMultiplier;
    float nearPressureMultiplier;
    float viscosityStrength;
    vec2 boundsSize;
    vec2 obstacleSize;
    vec2 obstacleCentre;
    uint numParticles;
    uint kernelIndex;
    vec2 interactionInputPoint;
    float interactionInputStrength;
    float interactionInputRadius;
    float Poly6ScalingFactor;
    float SpikyPow3ScalingFactor;
    float SpikyPow2ScalingFactor;
    float SpikyPow3DerivativeScalingFactor;
    float SpikyPow2DerivativeScalingFactor;
    float _padding1;
    float _padding2;
    float _padding3;
} pc;

// ========== Storage Buffers ==========
layout(set = 0, binding = 0, std430) restrict buffer PositionBuffer {
    vec2 positions[];
} positionBuffer;

layout(set = 0, binding = 1, std430) restrict buffer PredictedPositionBuffer {
    vec2 predictedPositions[];
} predictedPositionBuffer;

layout(set = 0, binding = 2, std430) restrict buffer VelocityBuffer {
    vec2 velocities[];
} velocityBuffer;

layout(set = 0, binding = 3, std430) restrict buffer DensityBuffer {
    vec2 densities[];
} densityBuffer;

layout(set = 0, binding = 4, std430) restrict buffer SpatialIndicesBuffer {
    SpatialEntry spatialIndices[];
} spatialIndicesBuffer;

layout(set = 0, binding = 5, std430) restrict buffer SpatialOffsetsBuffer {
    uint spatialOffsets[];
} spatialOffsetsBuffer;

// ========== Smoothing Kernel Functions ==========
float SmoothingKernelPoly6(float dst, float radius) {
    if (dst < radius) {
        float v = radius * radius - dst * dst;
        return v * v * v * pc.Poly6ScalingFactor;
    }
    return 0.0;
}

float SpikyKernelPow3(float dst, float radius) {
    if (dst < radius) {
        float v = radius - dst;
        return v * v * v * pc.SpikyPow3ScalingFactor;
    }
    return 0.0;
}

float SpikyKernelPow2(float dst, float radius) {
    if (dst < radius) {
        float v = radius - dst;
        return v * v * pc.SpikyPow2ScalingFactor;
    }
    return 0.0;
}

float DerivativeSpikyPow3(float dst, float radius) {
    if (dst <= radius) {
        float v = radius - dst;
        return -v * v * pc.SpikyPow3DerivativeScalingFactor;
    }
    return 0.0;
}

float DerivativeSpikyPow2(float dst, float radius) {
    if (dst <= radius) {
        float v = radius - dst;
        return -v * pc.SpikyPow2DerivativeScalingFactor;
    }
    return 0.0;
}

// ========== Spatial Hashing Functions ==========
ivec2 GetCell2D(vec2 position, float radius) {
    return ivec2(floor(position / radius));
}

uint HashCell2D(ivec2 cell) {
    uvec2 ucell = uvec2(cell);
    uint a = ucell.x * hashK1;
    uint b = ucell.y * hashK2;
    return a + b;
}

uint KeyFromHash(uint hash, uint tableSize) {
    return hash % tableSize;
}

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;


// ========== Compute Shader Kernels ==========

// External forces such as gravity and interaction input
// input is currently broken, not sure why, but its a low priority
void ExternalForces(uint id) {
    vec2 pos = positionBuffer.positions[id];
    vec2 vel = velocityBuffer.velocities[id];
    vec2 gravityAccel = vec2(0.0, pc.gravity);

    // input interaction
    if (pc.interactionInputStrength != 0.0) {
        vec2 inputPointOffset = pc.interactionInputPoint - pos;
        float sqrDst = dot(inputPointOffset, inputPointOffset);

        if (sqrDst < pc.interactionInputRadius * pc.interactionInputRadius) {
            float dst = sqrt(sqrDst);
            float edgeT = dst / pc.interactionInputRadius;
            float centreT = 1.0 - edgeT;
            vec2 dirToCentre = inputPointOffset / dst;

            float gravityWeight = 1.0 - (centreT * clamp(pc.interactionInputStrength / 10.0, 0.0, 1.0));
            vec2 accel = gravityAccel * gravityWeight + dirToCentre * centreT * pc.interactionInputStrength;
            accel -= vel * centreT;
            gravityAccel = accel;
        }
    }

    velocityBuffer.velocities[id] += gravityAccel * pc.deltaTime;

    const float predictionFactor = 1.0 / 120.0;
    predictedPositionBuffer.predictedPositions[id] = pos + velocityBuffer.velocities[id] * predictionFactor;
}

void UpdateSpatialHash(uint id) {
    // reset offset
    spatialOffsetsBuffer.spatialOffsets[id] = pc.numParticles;
    
    // calculate spatial hash
    vec2 pos = predictedPositionBuffer.predictedPositions[id];
    ivec2 cell = GetCell2D(pos, pc.smoothingRadius);
    uint hash = HashCell2D(cell);
    uint key = KeyFromHash(hash, pc.numParticles);
    
    // populate spatial indices
    spatialIndicesBuffer.spatialIndices[id].originalIndex = id;
    spatialIndicesBuffer.spatialIndices[id].hash = hash;
    spatialIndicesBuffer.spatialIndices[id].key = key;
    spatialIndicesBuffer.spatialIndices[id]._padding = 0;
}

void CalculateDensities(uint id) {
    vec2 pos = predictedPositionBuffer.predictedPositions[id];
    ivec2 originCell = GetCell2D(pos, pc.smoothingRadius);
    float sqrRadius = pc.smoothingRadius * pc.smoothingRadius;
    float density = 0.0;
    float nearDensity = 0.0;

    // density += SpikyKernelPow2(0.0, pc.smoothingRadius);
    // nearDensity += SpikyKernelPow3(0.0, pc.smoothingRadius);

    // neighbor search
    for (int i = 0; i < 9; i++) {
        uint hash = HashCell2D(originCell + offsets2D[i]);
        uint key = KeyFromHash(hash, pc.numParticles);
        uint currentIndex = spatialOffsetsBuffer.spatialOffsets[key];

        if (currentIndex >= pc.numParticles) continue;

        while (currentIndex < pc.numParticles) {
            SpatialEntry indexData = spatialIndicesBuffer.spatialIndices[currentIndex];
            currentIndex++;

            if (indexData.key != key) break;
            if (indexData.hash != hash) continue;

            uint neighborIndex = indexData.originalIndex;
            // in the case of density, we actually do want the self contribution, so no skipping here
            // if (neighborIndex == id) continue;

            vec2 neighborPos = predictedPositionBuffer.predictedPositions[neighborIndex];
            vec2 offsetToNeighbor = neighborPos - pos;
            float sqrDstToNeighbor = dot(offsetToNeighbor, offsetToNeighbor);

            if (sqrDstToNeighbor > sqrRadius) continue;

            float dst = sqrt(sqrDstToNeighbor);
            density += SpikyKernelPow2(dst, pc.smoothingRadius);
            nearDensity += SpikyKernelPow3(dst, pc.smoothingRadius);
        }
    }

    densityBuffer.densities[id] = vec2(density, nearDensity);
}

void CalculatePressureForce(uint id) {
    vec2 densityData = densityBuffer.densities[id];
    float density = densityData.x;
    float nearDensity = densityData.y;

    if (density <= 0.0 || nearDensity <= 0.0) {
        return;
    }

    float pressure = (density - pc.targetDensity) * pc.pressureMultiplier;
    float nearPressure = nearDensity * pc.nearPressureMultiplier;

    vec2 pressureForce = vec2(0.0);
    vec2 pos = predictedPositionBuffer.predictedPositions[id];
    ivec2 originCell = GetCell2D(pos, pc.smoothingRadius);
    float sqrRadius = pc.smoothingRadius * pc.smoothingRadius;

    // neighbor search
    for (int i = 0; i < 9; i++) {
        uint hash = HashCell2D(originCell + offsets2D[i]);
        uint key = KeyFromHash(hash, pc.numParticles);
        uint currentIndex = spatialOffsetsBuffer.spatialOffsets[key];

        if (currentIndex >= pc.numParticles) continue;

        while (currentIndex < pc.numParticles) {
            SpatialEntry indexData = spatialIndicesBuffer.spatialIndices[currentIndex];
            currentIndex++;

            if (indexData.key != key) break;
            if (indexData.hash != hash) continue;

            uint neighborIndex = indexData.originalIndex;
            if (neighborIndex == id) continue; // skip self contribution for pressure force

            vec2 neighborPos = predictedPositionBuffer.predictedPositions[neighborIndex];
            vec2 offsetToNeighbor = neighborPos - pos;
            float sqrDstToNeighbor = dot(offsetToNeighbor, offsetToNeighbor);

            if (sqrDstToNeighbor > sqrRadius) continue;

            float dst = sqrt(sqrDstToNeighbor);
            vec2 dirToNeighbor = dst > 0 ? offsetToNeighbor / dst : vec2(0, 1);

            vec2 neighborDensityData = densityBuffer.densities[neighborIndex];
            float neighborDensity = max(neighborDensityData.x, 0.0001);
            float neighborNearDensity = max(neighborDensityData.y, 0.0001);

            float neighborPressure = (neighborDensity - pc.targetDensity) * pc.pressureMultiplier;
            float neighborNearPressure = neighborNearDensity * pc.nearPressureMultiplier;

            float sharedPressure = (pressure + neighborPressure) * 0.5;
            float sharedNearPressure = (nearPressure + neighborNearPressure) * 0.5;

            pressureForce += dirToNeighbor * DerivativeSpikyPow2(dst, pc.smoothingRadius) * sharedPressure / neighborDensity;
            pressureForce += dirToNeighbor * DerivativeSpikyPow3(dst, pc.smoothingRadius) * sharedNearPressure / neighborNearDensity;
        }
    }

    vec2 acceleration = pressureForce / density;
    velocityBuffer.velocities[id] += acceleration * pc.deltaTime;
}

void CalculateViscosity(uint id) {
    vec2 pos = predictedPositionBuffer.predictedPositions[id];
    vec2 velocity = velocityBuffer.velocities[id];
    ivec2 originCell = GetCell2D(pos, pc.smoothingRadius);
    float sqrRadius = pc.smoothingRadius * pc.smoothingRadius;
    vec2 viscosityForce = vec2(0.0);
    
    // Neighbor search
    for (int i = 0; i < 9; i++) {
        uint hash = HashCell2D(originCell + offsets2D[i]);
        uint key = KeyFromHash(hash, pc.numParticles);
        uint currIndex = spatialOffsetsBuffer.spatialOffsets[key];
        
        while (currIndex < pc.numParticles) {
            SpatialEntry indexData = spatialIndicesBuffer.spatialIndices[currIndex];
            currIndex++;

            if (indexData.key != key) break;
            if (indexData.hash != hash) continue;

            uint neighbourIndex = indexData.originalIndex;
            if (neighbourIndex == id) continue;
            
            vec2 neighbourPos = predictedPositionBuffer.predictedPositions[neighbourIndex];
            vec2 offsetToNeighbour = neighbourPos - pos;
            float sqrDstToNeighbour = dot(offsetToNeighbour, offsetToNeighbour);
            
            if (sqrDstToNeighbour > sqrRadius) continue;
            
            float dst = sqrt(sqrDstToNeighbour);
            vec2 neighbourVelocity = velocityBuffer.velocities[neighbourIndex];
            viscosityForce += (neighbourVelocity - velocity) * SmoothingKernelPoly6(dst, pc.smoothingRadius);
        }
    }
    
    velocityBuffer.velocities[id] += viscosityForce * pc.viscosityStrength * pc.deltaTime;
}

void UpdatePositions(uint id) {
    vec2 vel = velocityBuffer.velocities[id];
    vec2 pos = positionBuffer.positions[id] + vel * pc.deltaTime;
    
    // Handle boundary collisions
    vec2 halfSize = pc.boundsSize * 0.5;
    vec2 edgeDst = halfSize - abs(pos);
    
    if (edgeDst.x <= 0) {
        pos.x = halfSize.x * sign(pos.x);
        vel.x *= -1.0 * pc.collisionDamping;
    }
    if (edgeDst.y <= 0) {
        pos.y = halfSize.y * sign(pos.y);
        vel.y *= -1.0 * pc.collisionDamping;
    }
    
    // Handle obstacle collision
    vec2 obstacleHalfSize = pc.obstacleSize * 0.5;
    vec2 obstacleEdgeDst = obstacleHalfSize - abs(pos - pc.obstacleCentre);
    
    if (obstacleEdgeDst.x >= 0 && obstacleEdgeDst.y >= 0) {
        if (obstacleEdgeDst.x < obstacleEdgeDst.y) {
            pos.x = obstacleHalfSize.x * sign(pos.x - pc.obstacleCentre.x) + pc.obstacleCentre.x;
            vel.x *= -1.0 * pc.collisionDamping;
        } else {
            pos.y = obstacleHalfSize.y * sign(pos.y - pc.obstacleCentre.y) + pc.obstacleCentre.y;
            vel.y *= -1.0 * pc.collisionDamping;
        }
    }
    
    positionBuffer.positions[id] = pos;
    velocityBuffer.velocities[id] = vel;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= pc.numParticles) return;
    
    // Execute kernel based on push constant
    switch(pc.kernelIndex) {
        case KERNEL_EXTERNAL_FORCES:
            ExternalForces(id);
            break;
        case KERNEL_SPATIAL_HASH:
            UpdateSpatialHash(id);
            break;
        case KERNEL_CALCULATE_DENSITIES:
            CalculateDensities(id);
            break;
        case KERNEL_PRESSURE_FORCE:
            CalculatePressureForce(id);
            break;
        case KERNEL_VISCOSITY:
            CalculateViscosity(id);
            break;
        case KERNEL_UPDATE_POSITIONS:
            UpdatePositions(id);
            break;
    }
}