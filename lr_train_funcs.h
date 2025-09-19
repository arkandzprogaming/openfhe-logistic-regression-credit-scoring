//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2023, Duality Technologies Inc.
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

#ifndef DPRIVE_ML_LR_TRAIN_FUNCS_H
#define DPRIVE_ML_LR_TRAIN_FUNCS_H

#include "lr_types.h"
#include "parameters.h"

///////////////////////////////////////
// Logistic Regression Training Functions
///////////////////////////////////////

Mat InitializeLogReg(Mat &X, Mat &y, float scalingFactor);

void EncLogRegCalculateGradient(
    CC &cc,
    const CT &ctX,
    const CT &ctNegXt,
    const CT &ctLabels,
    CT &ctThetas,
    CT &ctGradStoreInto,
    const usint rowSize,
    const MatKeys &rowKeys,
    const MatKeys &colKeys,
    const KeyPair &keys,
    bool debug = false,
    int chebRangeStart = -16,
    int chebRangeEnd = 16,
    int chebPolyDegree = 59,
    int debugPlaintextLength = 32
);

void BoundCheckMat(const Mat &inMat, const double bound);
PT ReEncrypt(CC &cc, CT &ctx, const KeyPair &keys);
int ReturnDepth(const CT &ct);
double ComputeLoss(const Mat &b, const Mat &X, const Mat &y);

///////////////////////////////////////
// Performance Metrics Functions
///////////////////////////////////////

// Generate binary predictions from probabilities using threshold
Mat MakePredictions(const Mat &probabilities, double threshold = 0.5);

// Generate probability predictions from features and weights
Mat ComputeProbabilities(const Mat &X, const Mat &weights);

// Calculate accuracy score
double ComputeAccuracy(const Mat &y_true, const Mat &y_pred);

// Calculate precision score
double ComputePrecision(const Mat &y_true, const Mat &y_pred);

// Calculate recall score
double ComputeRecall(const Mat &y_true, const Mat &y_pred);

// Calculate F1 score
double ComputeF1Score(const Mat &y_true, const Mat &y_pred);

// Calculate ROC AUC score
double ComputeROCAUC(const Mat &y_true, const Mat &y_prob);

// Structure to hold all performance metrics
struct PerformanceMetrics {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    double roc_auc;
    double loss;
};

// Compute all performance metrics at once
PerformanceMetrics ComputeAllMetrics(const Mat &weights, const Mat &X, const Mat &y, double threshold = 0.5);

#endif // DPRIVE_ML_LR_TRAIN_FUNCS_H
