#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "ext_cpp_get_evidence.h"

void get_evidence(size_t numCandidates,
                  size_t numPairs,
                  uint64_t* softMaskShape,
                  uint64_t* softMaskOffset,
                  const uint64_t* candidates, //positions of candidates (2d)
                  const uint64_t* pairs, //pairs of candidate ids (2d)
                  const double* softMaskArray, //voxel values (3d)
                  const uint32_t* voxelSize,
                  double* evidenceArr){

    uint64_t posU [3];
    uint64_t posV [3];
    int64_t diff [3];
    int64_t maxDirection [3];
    uint64_t absMaxDirection [3];
    int64_t maxLength;
    double stepVec [3];

    for (size_t i = 0; i < numPairs; i++) {
        uint64_t u = pairs[i*2];
        uint64_t v = pairs[i*2 + 1];

        // iterate over positions
        for (size_t k = 0; k < 3; k++){
            posU[k] = candidates[u*4 + k+1]/voxelSize[k];
            posV[k] = candidates[v*4 + k+1]/voxelSize[k];
            diff[k] = posV[k] - posU[k];
            maxDirection[k] = diff[k] * voxelSize[k];
            absMaxDirection[k] = std::abs(maxDirection[k]);
        }
 
        maxLength = *std::max_element(absMaxDirection, absMaxDirection + 3);

        for (size_t k = 0; k < 3; k++){
            stepVec[k] = 1.0 * maxDirection[k] / maxLength;
        }

        // Interpolate line
        uint64_t p_0 [3] = {posU[0] - softMaskOffset[0], 
                            posU[1] - softMaskOffset[1],
                            posU[2] - softMaskOffset[2]};
        double evidence = softMaskArray[p_0[2] + softMaskShape[2] * (p_0[1] + softMaskShape[1] * p_0[0])];
        uint64_t lenLine = 1;
        for (size_t step = 0; step<maxLength; step++) {
            uint64_t p [3];
            for (size_t k = 0; k < 3; k++){
                // This is a nasty but needed for consistency with prior implementation
                p[k] = static_cast <uint64_t> 
                       (
                            static_cast <uint64_t> 
                            (
                                (step + 1) * stepVec[k] + posU[k] * voxelSize[k] + 0.5
                            ) 
                            / (1.0 * voxelSize[k]) + 0.5
                        ) - softMaskOffset[k];
            }
            if (!((p[0] == p_0[0]) && (p[1] == p_0[1]) && (p[2] == p_0[2])))
            {
                evidence += softMaskArray[p[2] + softMaskShape[2] * (p[1] + softMaskShape[1] * p[0])];
                p_0[0] = p[0];
                p_0[1] = p[1];
                p_0[2] = p[2];
                lenLine += 1;
            }

        }
        
        evidence /= (lenLine*255.);
        evidenceArr[i*3] = 1.0 * candidates[4*u];
        evidenceArr[i*3+1] = 1.0 * candidates[4*v];
        evidenceArr[i*3+2] = evidence;
    }
}
