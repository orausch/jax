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

#include "jaxlib/tpu/tpu_dialect.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep.
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep.
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"
#include "jaxlib/tpu/tpu_dialect.cc.inc"
#include "jaxlib/tpu/tpu_enums.cc.inc"

// This is a bit unclean, but we need to squat the xla namespace to make sure
// that this overload is found via argument-dependent lookup.
namespace xla {

llvm::hash_code hash_value(const ::xla::Tile &p) { return absl::HashOf(p); }

}  // namespace xla

#define GET_ATTRDEF_CLASSES
#include "jaxlib/tpu/tpu_attr_defs.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "jaxlib/tpu/tpu_type_defs.cc.inc"

namespace mlir::tpu {

void TPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "jaxlib/tpu/tpu_attr_defs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "jaxlib/tpu/tpu_type_defs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "jaxlib/tpu/tpu_ops.cc.inc"
      >();
}

void VectorLayoutAttr::print(AsmPrinter &printer) const {
  printer << '<';
  printer << getLayout();
  printer << '>';
}

Attribute VectorLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  if (auto layout = parseLayout(parser);
      layout.has_value() && succeeded(parser.parseGreater())) {
    return get(parser.getContext(), *layout);
  }
  return {};
}

void TiledLayoutAttr::print(AsmPrinter &printer) const {
  printer << '<';
  printer << getRank();
  printer << ',';
  for (const xla::Tile &tile : getTiles()) {
    printer << tile.ToString();
  }
  printer << '>';
}

Attribute TiledLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  int64_t rank;
  if (parser.parseInteger(rank) || parser.parseComma()) {
    return {};
  }
  llvm::SmallVector<xla::Tile, 2> tiles;
  int64_t size;
  while (succeeded(parser.parseOptionalLParen())) {
    xla::Tile &tile = tiles.emplace_back();
    bool first = true;
    while (!succeeded(parser.parseOptionalRParen())) {
      if (!first) {
        if (failed(parser.parseComma())) {
          return {};
        }
      }
      first = false;
      if (failed(parser.parseInteger(size))) {
        return {};
      }
      tile.add_dimensions(size);
    }
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  return get(parser.getContext(), rank, tiles);
}

AffineMap TiledLayoutAttr::getAffineMap() const {
  AffineMap map = AffineMap::getMultiDimIdentityMap(getRank(), getContext());
  SmallVector<AffineExpr, 8> exprs;
  for (const xla::Tile &tile : getTiles()) {
    exprs.clear();
    auto dimensions = tile.dimensions();
    int64_t untiled_dims = map.getNumResults() - dimensions.size();
    if (untiled_dims < 0) {
      LOG(FATAL) << "Invalid TiledLayoutAttr!";
    }
    for (int64_t i = 0; i < untiled_dims; ++i) {
      exprs.push_back(getAffineDimExpr(i, getContext()));
    }
    for (int i = 0; i < dimensions.size(); ++i) {
      exprs.push_back(getAffineDimExpr(untiled_dims + i, getContext())
                          .floorDiv(dimensions[i]));
    }
    for (int i = 0; i < dimensions.size(); ++i) {
      exprs.push_back(getAffineDimExpr(untiled_dims + i, getContext()) %
                      dimensions[i]);
    }
    auto tile_map = AffineMap::get(map.getNumResults(), 0, exprs, getContext());
    map = tile_map.compose(map);
  }
  return map;
}

MemRefType getMemRefType(Value value) {
  if (auto erase_op = value.getDefiningOp<tpu::EraseLayoutOp>()) {
    value = erase_op.getOperand();
  }
  return cast<MemRefType>(value.getType());
}

}  // namespace mlir::tpu
