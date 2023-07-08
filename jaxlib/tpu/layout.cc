/* Copyright 2023 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/tpu/layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tpu {

namespace {

mlir::ParseResult parseOffset(llvm::StringRef* data,
                              std::optional<int64_t>* result) {
  int64_t int_result;
  if (data->consume_front("*")) {
    *result = std::nullopt;
    return success();
  }
  if (!data->consumeInteger(10, int_result)) {
    *result = int_result;
    return success();
  }
  return failure();
}

}  // namespace

std::tuple<std::optional<int64_t>, std::optional<int64_t>, int64_t, int64_t,
           int8_t, VectorLayout::ImplicitDim>
VectorLayout::as_tuple() const {
  return std::make_tuple(offsets_[0], offsets_[1], tiling_[0], tiling_[1],
                         bitwidth_, implicit_dim_);
}

bool VectorLayout::operator==(const VectorLayout& other) const {
  return as_tuple() == other.as_tuple();
}

template <typename Stream>
void VectorLayout::print(Stream& os) const {
  os << static_cast<int32_t>(bitwidth_) << ",{";
  bool first = true;
  for (auto o : offsets_) {
    if (first) {
      first = false;
    } else {
      os << ',';
    }
    if (!o) {
      os << '*';
    } else {
      os << *o;
    }
  }
  os << "},(" << tiling_[0] << ',' << tiling_[1] << ")";
  if (implicit_dim_ == ImplicitDim::kMinor) {
    os << ",-1";
  } else if (implicit_dim_ == ImplicitDim::kSecondMinor) {
    os << ",-2";
  }
}

std::optional<VectorLayout> VectorLayout::join(const VectorLayout& l,
                                               const VectorLayout& r,
                                               ArrayRef<int64_t> shape) {
  if (l.bitwidth_ != r.bitwidth_ || l.tiling_ != r.tiling_) {
    return std::nullopt;
  }
  if (l.implicit_dim_ != r.implicit_dim_) {
    if (shape.size() < 2) {
      return std::nullopt;
    }
    ImplicitDim dim;
    if (l.implicit_dim_ == ImplicitDim::kNone) {
      dim = r.implicit_dim_;
    } else if (r.implicit_dim_ == ImplicitDim::kNone) {
      dim = l.implicit_dim_;
    } else {
      return std::nullopt;
    }
    if (dim == ImplicitDim::kMinor && shape[shape.size() - 1] == 1) {
      // OK, they are equivalent.
    } else if (dim == ImplicitDim::kSecondMinor &&
               shape[shape.size() - 2] == 1) {
      // OK, they are equivalent.
    } else {
      return std::nullopt;
    }
  }
  LayoutOffsets offsets;
  for (int i = 0; i < 2; ++i) {
    auto lo = l.offsets()[i];
    auto ro = r.offsets()[i];
    if (lo && ro && lo != ro) {
      return std::nullopt;
    }
    offsets[i] = lo.has_value() ? lo : ro;
  }
  return VectorLayout(l.bitwidth_, offsets, l.tiling_, l.implicit_dim_);
}

std::optional<VectorLayout> VectorLayout::parse(llvm::StringRef* data) {
  llvm::StringRef local(*data);
  int8_t bitwidth;
  LayoutOffsets offsets;
  std::array<int64_t, 2> tiling;
  ImplicitDim implicit_dim = ImplicitDim::kNone;
  if (local.consumeInteger(10, bitwidth) || !local.consume_front(",{") ||
      parseOffset(&local, &offsets[0]) || !local.consume_front(",") ||
      parseOffset(&local, &offsets[1]) || !local.consume_front("},(") ||
      local.consumeInteger(10, tiling[0]) || !local.consume_front(",") ||
      local.consumeInteger(10, tiling[1]) || !local.consume_front(")")) {
    return std::nullopt;
  }
  if (local.consume_front(",-1")) {
    implicit_dim = ImplicitDim::kMinor;
  } else if (local.consume_front(",-2")) {
    implicit_dim = ImplicitDim::kSecondMinor;
  }
  *data = local;
  return VectorLayout(bitwidth, offsets, tiling, implicit_dim);
}

namespace {
template<class> inline constexpr bool false_v = false;

template <typename Stream>
Stream& printLayout(Stream& os, const Layout& v) {
  os << '"';
  if (v.has_value()) {
    v->print(os);
  } else {
    os << "none";
  }
  os << '"';
  return os;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const Layout& v) {
  return printLayout<std::ostream>(os, v);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Layout& v) {
  return printLayout<llvm::raw_ostream>(os, v);
}

llvm::hash_code hash_value(const VectorLayout& layout) {
  return llvm::hash_value(layout.as_tuple());
}

std::optional<Layout> parseLayout(mlir::AsmParser& parser) {
  std::string layout_str;
  if (failed(parser.parseString(&layout_str))) {
    return std::nullopt;
  }
  if (layout_str == "none") {
    return kNoLayout;
  }
  llvm::StringRef ref(layout_str);
  if (auto layout = VectorLayout::parse(&ref); ref.empty()) {
    return *layout;
  }
  return std::nullopt;
}

const Layout kNoLayout = std::nullopt;

}  // namespace mlir::tpu
