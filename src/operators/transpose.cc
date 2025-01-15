#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    auto input_dim = A->getDims();
    auto output_dim = input_dim;
    int rank = A->getRank();

    // 验证permute向量的合法性
    if (!transposePermute.empty()) {
        // 1. 检查permute大小是否匹配维度数
        IT_ASSERT(static_cast<int>(transposePermute.size()) == rank);
        
        // 2. 检查permute中的值是否有效（是否在[0, rank-1]范围内且不重复）
        vector<bool> used(rank, false);
        for (int i = 0; i < rank; ++i) {
            IT_ASSERT(transposePermute[i] >= 0 && transposePermute[i] < rank);
            IT_ASSERT(!used[transposePermute[i]]);
            used[transposePermute[i]] = true;
        }

        // 3. 根据permute重排维度
        for (int i = 0; i < rank; ++i) {
            output_dim[i] = input_dim[transposePermute[i]];
        }
    } else {
        // 如果permute为空，默认反转所有维度
        for (int i = 0; i < rank; ++i) {
            output_dim[i] = input_dim[rank - 1 - i];
        }
    }

    return {{output_dim}};
}

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
