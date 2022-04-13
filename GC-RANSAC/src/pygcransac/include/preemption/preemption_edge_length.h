// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)

// This file created 2021 by Amnon Drory, Tel-Aviv University

#pragma once

#include "model.h"
#include <opencv2/core.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
using std::cout;
using std::endl;
using std::sqrt;

namespace gcransac
{
	namespace preemption
	{
		template <typename _ModelEstimator>
		class EdgeLenPreemptiveVerification
		{

		public:


			static constexpr bool providesScore() { return false; }
			static constexpr const char *getName() { return "EdgeLen"; }

			EdgeLenPreemptiveVerification(const cv::Mat &points_,
				const _ModelEstimator &estimator_,
				const double &minimum_inlier_ratio_ = 0.1)
			{
			}

			bool verifyModel(const gcransac::Model &model_,
				const _ModelEstimator &estimator_, // The model estimator
				const double &threshold_,
				const size_t &iteration_number_,
				const Score &best_score_,
				const cv::Mat &points_,
				const size_t *minimal_sample_,
				const size_t sample_number_,
				std::vector<size_t> &inliers_,
				Score &score_)
			{
				const double SIMILARITY_THRESHOLD = 0.9;

				const double *data_ptr = reinterpret_cast<double *>(points_.data);
				const int cols = points_.cols;
				
				for (size_t i = 0; i < sample_number_; i++)
				{
					size_t i_offset = cols * minimal_sample_[i];
					
					const double &i_src_x = data_ptr[i_offset];
					const double &i_src_y = data_ptr[i_offset + 1];
					const double &i_src_z = data_ptr[i_offset + 2];
					const double &i_tgt_x = data_ptr[i_offset + 3];
					const double &i_tgt_y = data_ptr[i_offset + 4];
					const double &i_tgt_z = data_ptr[i_offset + 5];
					
					for (size_t j = i+1; j < sample_number_; j++)
					{
						size_t j_offset = cols * minimal_sample_[j];
						
						const double &j_src_x = data_ptr[j_offset];
						const double &j_src_y = data_ptr[j_offset + 1];
						const double &j_src_z = data_ptr[j_offset + 2];
						const double &j_tgt_x = data_ptr[j_offset + 3];
						const double &j_tgt_y = data_ptr[j_offset + 4];
						const double &j_tgt_z = data_ptr[j_offset + 5];		
						
						const double d_src_x = j_src_x - i_src_x;
						const double d_src_y = j_src_y - i_src_y;
						const double d_src_z = j_src_z - i_src_z;
						const double d_tgt_x = j_tgt_x - i_tgt_x;
						const double d_tgt_y = j_tgt_y - i_tgt_y;
						const double d_tgt_z = j_tgt_z - i_tgt_z;
						
						const double dist_src = sqrt(d_src_x*d_src_x + d_src_y*d_src_y + d_src_z*d_src_z);
						const double dist_tgt = sqrt(d_tgt_x*d_tgt_x + d_tgt_y*d_tgt_y + d_tgt_z*d_tgt_z);
										
						if ( ( dist_src < (dist_tgt * SIMILARITY_THRESHOLD) ) ||
							 ( dist_tgt < (dist_src * SIMILARITY_THRESHOLD) ) )
						{
							return false;
						}
					}	
				}
				
				return true;
			}
		};
	}
}
