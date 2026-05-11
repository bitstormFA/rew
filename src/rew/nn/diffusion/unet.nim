## U-Net building blocks for diffusion models.
##
## Provides ResBlock, DownBlock, UpBlock, MidBlock, and a configurable
## UNet. All layers are pure value types following rew's functional nn
## invariant. Spatial self-attention is applied by reshaping NHWC
## features to sequence format and using the existing MultiHeadAttention.
##
## The U-Net expects NHWC inputs and produces NHWC outputs. Time
## conditioning is injected into every ResBlock via a learned
## scale + shift projection.

import ../../tensor
import ../../rng
import ../../ops/arith
import ../../ops/literal
import ../../ops/shape
import ../../ops/linalg
import ../../ops/concat
import ../norm
import ../conv
import ../linear
import ../activation
import ../multiheadattention
import ../upsample
import ./time_embed

# ---- ResBlock ----------------------------------------------------------------

type
  ResBlock* = object
    ## Residual block with GroupNorm, SiLU activation, and time conditioning.
    ##
    ## Forward:  h = GroupNorm(x) → SiLU → Conv(inCh, outCh)
    ##           h = h + timeProj(timeEmbed)  (broadcast as scale+shift)
    ##           h = GroupNorm(h) → SiLU → Conv(outCh, outCh)
    ##           out = skipConv(x) + h
    norm1*: GroupNorm
    norm2*: GroupNorm
    conv1*: Conv2d
    conv2*: Conv2d
    timeProj*: Linear             ## timeEmbedDim → 2 * outChannels
    skipConv*: Conv2d             ## 1x1 conv for channel-matching skip
    inChannels*: int
    outChannels*: int

proc initResBlock*(key: Key; inChannels, outChannels, timeEmbedDim: int;
    numGroups: int = 32): ResBlock =
  ## Constructs a ResBlock.
  ##
  ## Uses 3×3 convolutions with SAME-like padding. The time projection
  ## outputs `2 * outChannels` values: first half is scale, second half
  ## is shift, both applied after the first GroupNorm+SiLU.
  let keys = split(key, 4)
  ResBlock(
    norm1: initGroupNorm(numGroups, inChannels),
    norm2: initGroupNorm(numGroups, outChannels),
    conv1: initConv2d(keys[0], inChannels, outChannels,
      [3, 3], padding = [[1, 1], [1, 1]]),
    conv2: initConv2d(keys[1], outChannels, outChannels,
      [3, 3], padding = [[1, 1], [1, 1]]),
    timeProj: initLinear(keys[2], timeEmbedDim, 2 * outChannels),
    skipConv: initConv2d(keys[3], inChannels, outChannels, [1, 1]),
    inChannels: inChannels,
    outChannels: outChannels,
  )

proc forward*(layer: ResBlock; x, timeEmbed: Tensor): Tensor =
  ## Forward pass with time conditioning.
  ##
  ## `x`: `[N, H, W, inChannels]`, `timeEmbed`: `[N, timeEmbedDim]`.
  ## Returns: `[N, H, W, outChannels]`.
  # First norm + conv + time injection.
  var h = forward(layer.norm1, x)
  h = silu(h)
  h = forward(layer.conv1, h)
  # Time conditioning: project timeEmbed → [N, 2*outCh], split into
  # scale and shift, broadcast to NHWC, apply: h = h * (1 + scale) + shift.
  let timeOut = forward(layer.timeProj, timeEmbed)  # [N, 2*outCh]
  let halfCh = layer.outChannels
  let scaleRaw = slice(timeOut, @[0, 0], @[timeOut.shape[0], halfCh],
    @[1, 1])
  let shift = slice(timeOut, @[0, halfCh],
    @[timeOut.shape[0], 2 * halfCh], @[1, 1])
  let one = scalarF32(1'f32)
  let scale = add(broadcastTo(one, scaleRaw.shape, @[]), scaleRaw)
  let scaleB = broadcastTo(unsqueeze(unsqueeze(scale, 1), 1),
    h.shape, @[0, 3])
  let shiftB = broadcastTo(unsqueeze(unsqueeze(shift, 1), 1),
    h.shape, @[0, 3])
  h = add(mul(h, scaleB), shiftB)
  # Second norm + conv.
  h = forward(layer.norm2, h)
  h = silu(h)
  h = forward(layer.conv2, h)
  # Skip connection.
  let skip = forward(layer.skipConv, x)
  add(skip, h)

# ---- Spatial self-attention --------------------------------------------------

proc spatialSelfAttention*(x: Tensor; attn: MultiHeadAttention): Tensor =
  ## Apply self-attention to NHWC spatial features.
  ##
  ## Reshapes `[N, H, W, C]` → `[N, H*W, C]`, applies multi-head
  ## self-attention, then reshapes back to `[N, H, W, C]`.
  let n = x.shape[0]
  let h = x.shape[1]
  let w = x.shape[2]
  let c = x.shape[3]
  let seq = reshape(x, @[n, h * w, c])
  let attnOut = forward(attn, seq, seq, seq)
  reshape(attnOut, @[n, h, w, c])

# ---- DownBlock ---------------------------------------------------------------

type
  DownBlock* = object
    ## Downsampling block with ResBlocks, optional attention, and a
    ## stride-2 convolution for spatial reduction.
    res1*: ResBlock
    res2*: ResBlock
    attn*: MultiHeadAttention    ## Optional spatial self-attention
    hasAttn*: bool
    downsampler*: Conv2d          ## Stride-2 conv for 2× downsampling

proc initDownBlock*(key: Key; inChannels, outChannels, timeEmbedDim: int;
    hasAttn: bool = false; numHeads: int = 4): DownBlock =
  ## Constructs a DownBlock.
  ##
  ## Two ResBlocks followed by an optional attention layer and a stride-2
  ## conv for 2× spatial downsampling.
  let keys = split(key, int(3 + ord(hasAttn)))
  var ki = 0
  result = DownBlock(
    res1: initResBlock(keys[ki], inChannels, outChannels, timeEmbedDim),
    res2: initResBlock(keys[ki + 1], outChannels, outChannels, timeEmbedDim),
    hasAttn: hasAttn,
    downsampler: initConv2d(keys[ki + 2], outChannels, outChannels,
      [3, 3], stride = [2, 2], padding = [[1, 1], [1, 1]]),
  )
  if hasAttn:
    result.attn = initMultiHeadAttention(keys[ki + 3], outChannels, numHeads)

proc forward*(layer: DownBlock; x, timeEmbed: Tensor): Tensor =
  ## Forward pass: ResBlock → ResBlock → (attention) → downsample.
  var h = forward(layer.res1, x, timeEmbed)
  h = forward(layer.res2, h, timeEmbed)
  if layer.hasAttn:
    h = spatialSelfAttention(h, layer.attn)
  forward(layer.downsampler, h)

# ---- UpBlock -----------------------------------------------------------------

type
  UpBlock* = object
    ## Upsampling block with ResBlocks, optional attention, a skip
    ## connection from the corresponding DownBlock, and a bilinear
    ## upsampler.
    res1*: ResBlock
    res2*: ResBlock
    res3*: ResBlock
    attn*: MultiHeadAttention    ## Optional spatial self-attention
    hasAttn*: bool
    upsampler*: Upsample          ## 2× bilinear upsampling

proc initUpBlock*(key: Key; inChannels, outChannels, timeEmbedDim: int;
    hasAttn: bool = false; numHeads: int = 4): UpBlock =
  ## Constructs an UpBlock. `inChannels` includes the skip connection
  ## channels (typically 2× the output of the previous layer).
  let keys = split(key, int(4 + ord(hasAttn)))
  var ki = 0
  result = UpBlock(
    res1: initResBlock(keys[ki], inChannels, outChannels, timeEmbedDim),
    res2: initResBlock(keys[ki + 1], outChannels, outChannels, timeEmbedDim),
    res3: initResBlock(keys[ki + 2], outChannels, outChannels, timeEmbedDim),
    hasAttn: hasAttn,
    upsampler: initUpsample([2, 2], "bilinear"),
  )
  if hasAttn:
    result.attn = initMultiHeadAttention(keys[ki + 3], outChannels, numHeads)

proc forward*(layer: UpBlock; x, skip, timeEmbed: Tensor): Tensor =
  ## Forward pass: upsample → concat skip → ResBlocks → (attention).
  ##
  ## `x`: `[N, H, W, inCh]`, `skip`: `[N, 2*H, 2*W, skipCh]`,
  ## `timeEmbed`: `[N, timeEmbedDim]`.
  var h = forward(layer.upsampler, x)
  # Concatenate skip connection along channel dimension.
  h = concat(@[h, skip], 3)
  h = forward(layer.res1, h, timeEmbed)
  h = forward(layer.res2, h, timeEmbed)
  h = forward(layer.res3, h, timeEmbed)
  if layer.hasAttn:
    h = spatialSelfAttention(h, layer.attn)
  h

# ---- MidBlock ----------------------------------------------------------------

type
  MidBlock* = object
    ## Bottleneck block: ResBlock → Attention → ResBlock.
    res1*: ResBlock
    attn*: MultiHeadAttention
    res2*: ResBlock

proc initMidBlock*(key: Key; channels, timeEmbedDim: int;
    numHeads: int = 4): MidBlock =
  ## Constructs a MidBlock.
  let keys = split(key, 3)
  MidBlock(
    res1: initResBlock(keys[0], channels, channels, timeEmbedDim),
    attn: initMultiHeadAttention(keys[1], channels, numHeads),
    res2: initResBlock(keys[2], channels, channels, timeEmbedDim),
  )

proc forward*(layer: MidBlock; x, timeEmbed: Tensor): Tensor =
  ## Forward pass: ResBlock → Attention → ResBlock.
  var h = forward(layer.res1, x, timeEmbed)
  h = spatialSelfAttention(h, layer.attn)
  forward(layer.res2, h, timeEmbed)

# ---- UNet --------------------------------------------------------------------

type
  UNet* = object
    ## Full U-Net for diffusion models.
    ##
    ## Architecture: conv_in → {down blocks} → mid → {up blocks} → conv_out.
    ## Each "level" has channel count = baseChannels * chMult[i].
    ## Skip connections flow from down blocks to corresponding up blocks.
    timeEmbed*: TimeEmbedding
    convIn*: Conv2d
    downBlocks*: seq[DownBlock]
    midBlock*: MidBlock
    upBlocks*: seq[UpBlock]
    convOut*: Conv2d
    baseChannels*: int
    chMults*: seq[int]

proc initUNet*(key: Key; inChannels, outChannels, baseChannels: int;
    chMults: openArray[int]; timeEmbedDim: int = 128;
    attnResolutions: openArray[int] = @[16];
    numHeads: int = 4; numResBlocks: int = 2): UNet =
  ## Constructs a U-Net.
  ##
  ## `inChannels` / `outChannels`: input/output channel count.
  ## `baseChannels`: base channel count (e.g. 64 or 128).
  ## `chMults`: channel multipliers per level (e.g. [1, 2, 4, 4]).
  ## `timeEmbedDim`: dimension of the time embedding.
  ## `attnResolutions`: spatial resolutions at which to apply attention.
  ## `numHeads`: number of attention heads.
  ## `numResBlocks`: number of ResBlocks per down/up block.
  if inChannels <= 0 or outChannels <= 0 or baseChannels <= 0:
    raise newException(TensorError,
      "initUNet: channel counts must be positive")
  if chMults.len == 0:
    raise newException(TensorError,
      "initUNet: chMults must be non-empty")
  let numLevels = chMults.len
  var curKeys = split(key, 3 + numLevels * 2)
  var ki = 0
  let timeEmbed = initTimeEmbedding(curKeys[ki],
    timeEmbedDim * 4, timeEmbedDim * 4)
  inc ki
  let convIn = initConv2d(curKeys[ki], inChannels, baseChannels,
    [3, 3], padding = [[1, 1], [1, 1]])
  inc ki
  # Build down blocks.
  var downBlocks: seq[DownBlock] = @[]
  var ch = baseChannels
  var resolutions: seq[int] = @[]
  for level in 0 ..< numLevels:
    let outCh = baseChannels * chMults[level]
    for _ in 0 ..< numResBlocks:
      let hasAttn = attnResolutions.len > 0 and attnResolutions.contains(
        if resolutions.len > 0: resolutions[^1] else: 0)
      downBlocks.add initDownBlock(curKeys[ki], ch, outCh, timeEmbedDim * 4,
        hasAttn, numHeads)
      inc ki
      ch = outCh
    if level < numLevels - 1:
      resolutions.add -1
  # Mid block.
  let midBlock = initMidBlock(curKeys[ki], ch, timeEmbedDim * 4, numHeads)
  inc ki
  # Build up blocks (reverse channel order).
  var upBlocks: seq[UpBlock] = @[]
  for level in countdown(numLevels - 1, 0):
    let outCh = baseChannels * chMults[level]
    let skipCh = outCh
    let inCh = ch + skipCh
    for _ in 0 ..< numResBlocks:
      let hasAttn = attnResolutions.len > 0 and
        attnResolutions.contains(if resolutions.len > 0: resolutions[^1] else: 0)
      upBlocks.add initUpBlock(curKeys[ki], inCh, outCh,
        timeEmbedDim * 4, hasAttn, numHeads)
      inc ki
      ch = outCh
  let convOut = initConv2d(curKeys[ki], baseChannels, outChannels,
    [3, 3], padding = [[1, 1], [1, 1]])
  UNet(
    timeEmbed: timeEmbed,
    convIn: convIn,
    downBlocks: downBlocks,
    midBlock: midBlock,
    upBlocks: upBlocks,
    convOut: convOut,
    baseChannels: baseChannels,
    chMults: @chMults,
  )

proc forward*(model: UNet; x, t: Tensor): Tensor =
  ## Forward pass.
  ##
  ## `x`: `[N, H, W, inChannels]`, `t`: `[N]` float32 timesteps in [0, 1].
  ## Returns: `[N, H, W, outChannels]`.
  let timeEmb = forward(model.timeEmbed, t)
  var h = forward(model.convIn, x)
  # Store skip connections from down path.
  var skips: seq[Tensor] = @[]
  for blk in model.downBlocks:
    h = forward(blk, h, timeEmb)
    skips.add h
  h = forward(model.midBlock, h, timeEmb)
  # Up path with skip connections.
  for i in 0 ..< model.upBlocks.len:
    let skipIdx = skips.len - 1 - i
    h = forward(model.upBlocks[i], h, skips[skipIdx], timeEmb)
  forward(model.convOut, h)
