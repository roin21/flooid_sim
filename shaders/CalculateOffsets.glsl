#[compute]
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// sorted entry data
struct Entry {
    uint originalIndex;
    uint hash;
    uint key;
    uint _padding; // Padding to ensure alignment, if necessary
};

layout(push_constant) uniform PushConstants {
    uint numEntries;
} pc;

// inputs
layout(set = 0, binding = 0, std430) restrict readonly buffer EntriesBuffer {
    Entry entries[];
} entriesBuffer;

// outputs
layout(set = 0, binding = 1, std430) restrict buffer OffsetsBuffer {
    uint offsets[];
} offsetsBuffer;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= pc.numEntries) {
        return;
    }

    uint current_key = entriesBuffer.entries[id].key;

    // Check the key of the previous element to see if a new group starts here.
    // The first element (id=0) is always the start of a new group.
    uint previous_key = (id == 0) ? (current_key + 1) : entriesBuffer.entries[id - 1].key;

    // If the key is different from the previous one, this is the first particle of a new group.
    if (current_key != previous_key) {
        // needed to switch to atomic min to avoid potential race conditions when multiple threads try to write to the same offset
        atomicMin(offsetsBuffer.offsets[current_key], id);
    }
}