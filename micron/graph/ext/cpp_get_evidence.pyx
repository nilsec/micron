#cython: language_level=3
import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t, uint32_t

STUFF = "Hi"

cdef extern from "ext_cpp_get_evidence.h":
    void get_evidence(size_t numCandidates,
                      size_t numPairs,
                      uint64_t* softMaskShape,
                      uint64_t* softMaskOffset,
                      const uint64_t* candidates,
                      const uint64_t* pairs,
                      const double* softMaskArray,
                      const uint32_t* voxelSize,
                      double* evidenceArr);

def cpp_get_evidence(np.ndarray[uint64_t, ndim=2] candidates,
                     np.ndarray[uint64_t, ndim=2] pairs,
                     np.ndarray[double, ndim=3] soft_mask_array,
                     np.ndarray[uint64_t, ndim=1] soft_mask_offset,
                     np.ndarray[uint32_t, ndim=1] voxel_size):

    
    cdef size_t numCandidates = candidates.shape[0]
    cdef size_t numPairs = pairs.shape[0]
    cdef np.ndarray[np.uint64_t, ndim=1] softMaskShape = np.array(np.shape(soft_mask_array), dtype=np.uint64)
   
    if not candidates.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous candidates array (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        candidates = np.ascontiguousarray(candidates)
    if not pairs.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous pairs array (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        pairs = np.ascontiguousarray(pairs) 
    if not soft_mask_array.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous soft_mask_array (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        soft_mask_array = np.ascontiguousarray(soft_mask_array)

    cdef np.ndarray[double, ndim=2] evidenceArr = np.zeros((pairs.shape[0], 3), dtype=np.float64) # u, v, evidence
    if not evidenceArr.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous evidence_array (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        evidenceArr = np.ascontiguousarray(evidenceArr)

    get_evidence(numCandidates,
                 numPairs,
                 &softMaskShape[0],
                 &soft_mask_offset[0],
                 &candidates[0, 0],
                 &pairs[0, 0],
                 &soft_mask_array[0, 0, 0],
                 &voxel_size[0],
                 &evidenceArr[0, 0])

    return evidenceArr
