#[compute]
#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint numEntries;
} pc;

layout(set = 0, binding = 0, std430) restrict buffer OffsetsBuffer {
    uint offsets[];
} offsetsBuffer;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= pc.numEntries) return;
    offsetsBuffer.offsets[id] = pc.numEntries;
}