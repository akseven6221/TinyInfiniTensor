#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {
    // 处理标量情况（空张量）
    if (A.empty()) {
        return B;
    }
    if (B.empty()) {
        return A;
    }

    // 获取两个形状的维度数
    const size_t rankA = A.size();
    const size_t rankB = B.size();
    
    // 结果的维度数是两个输入中较大的
    const size_t resultRank = std::max(rankA, rankB);
    Shape result(resultRank);

    // 从最右边的维度开始处理
    for (size_t i = 0; i < resultRank; ++i) {
        const size_t rA = (i < rankA) ? rankA - 1 - i : 0;  // A 中对应的维度索引
        const size_t rB = (i < rankB) ? rankB - 1 - i : 0;  // B 中对应的维度索引

        const size_t dimA = (i < rankA) ? A[rA] : 1;
        const size_t dimB = (i < rankB) ? B[rB] : 1;

        // 广播规则：
        // 1. 如果维度相等，使用该维度
        // 2. 如果一个维度为1，使用另一个维度
        // 3. 如果两个维度都不为1且不相等，报错
        if (dimA == dimB) {
            result[resultRank - 1 - i] = dimA;
        } else if (dimA == 1) {
            result[resultRank - 1 - i] = dimB;
        } else if (dimB == 1) {
            result[resultRank - 1 - i] = dimA;
        } else {
            // 维度不兼容，无法广播
            IT_ASSERT(false);
            return {};
        }
    }

    return result;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
