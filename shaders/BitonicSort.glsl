#[compute]
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct Entry {
    uint originalIndex;
    uint hash;
    uint key;
    uint _padding;
};

layout(push_constant) uniform PushConstants {
    uint numEntries; // power-of-two size we are sorting
    uint k;          // size of the bitonic sequences to merge
    uint j;          // distance between elements to compare-and-swap
} pc;

layout(set = 0, binding = 0, std430) restrict buffer EntriesBuffer {
    Entry entries[];
} entriesBuffer;

// ngl, this is a weird algorithm and I do not fully understand it, nor do I know if I implemented it properly.
// you'd think i would figure it out since the entire simulation depends on it, but here we are.

// heres an explanation and visualization: https://www.youtube.com/watch?v=rSKMYc1CQHE (go to 41:15)
void main() {
    uint i = gl_GlobalInvocationID.x;
    
    // process pairs
    uint ixj = i ^ pc.j;
    
    // only process if we're the lower index of the pair
    if (ixj > i && i < pc.numEntries && ixj < pc.numEntries) {
        // can find the sort direction based on the current stage
        bool ascending = ((i & pc.k) == 0);
        
        // get the keys to compare
        uint key1 = entriesBuffer.entries[i].key;
        uint key2 = entriesBuffer.entries[ixj].key;
        
        // compare and swap if needed
        bool shouldSwap = (key1 > key2) == ascending;
        
        if (shouldSwap) {
            Entry temp = entriesBuffer.entries[i];
            entriesBuffer.entries[i] = entriesBuffer.entries[ixj];
            entriesBuffer.entries[ixj] = temp;
        }
    }
}