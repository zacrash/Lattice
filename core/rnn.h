#ifndef LATTICE_CORE_RNN_H
#define LATTICE_CORE_RNN_H

#include <Eigen/Dense>

using Eigen::MatrixXd;

namespace RNN {

    class RNN {
        // Model Parameters
        MatrixXd Wxh_;
        MatrixXd Whh_;
        MatrixXd Why_;
        VectorXd bh_;
        VectorXd by_;

        // Hyperparameters
        int hidden_size_;
        int seq_len_;
        float learning_rate_;

        public:
        RNN(int input_size, int hidden_size, int output_size) :
            hidden_size_(hidden_size),
            input_size_(input_size),
            output_size(output_size) {

            // Initialize parameters
            Wxh = MatrixXd::Random(hidden_size, input_size) * 0.01;
            Whh = MatrixXd::Random(hidden_size, hidden_size) * 0.01;
            Why = MatrixXd::Random(output_size, hidden_size) * 0.01;
            bh = VectorXd::Zero(hidden_size);
            by = VectorXd::Zero(output_size);

            // Default hyperparameters
            seq_len_ = 25;
            learning_rate_ = 0.01;
        }
    };
};
