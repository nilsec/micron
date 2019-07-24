void get_evidence(size_t numCandidates,
                  size_t numPairs,
                  uint64_t* softMaskShape,
                  uint64_t* softMaskOffset,
                  const uint64_t* candidates,
                  const uint64_t* pairs,
                  const double* softMaskArray,
                  const uint32_t* voxelSize,
                  double* evidenceArr);
