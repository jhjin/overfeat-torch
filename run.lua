require 'nn'
require 'image'
local ParamBank = require 'ParamBank'
local label     = require 'overfeat_label'
torch.setdefaulttensortype('torch.FloatTensor')


local opt = lapp([[
Parameters
   --network        (default 'small')      network size ( small | big )
   --img            (default 'bee.jpg')    test image
   --backend        (default 'nn')         specify backend ( nn | cunn | cudnn )
   --inplace                               use inplace ReLU
   --spatial                               use spatial mode (detection/localization)
   --save           (default 'model.t7')   save model path
   --threads        (default 4)            nb of threads
]])
torch.setnumthreads(opt.threads)
print('==> #threads:', torch.getnumthreads())
print('==> arguments')
print(opt)


-- set modules to be in use
if opt.backend == 'nn' or opt.backend == 'cunn' then
   require(opt.backend)
   SpatialConvolution = nn.SpatialConvolutionMM
   SpatialMaxPooling = nn.SpatialMaxPooling
   ReLU = nn.ReLU
   SpatialSoftMax = nn.SpatialSoftMax
elseif opt.backend == 'cudnn' then
   require(opt.backend)
   SpatialConvolution = cudnn.SpatialConvolution
   SpatialMaxPooling = cudnn.SpatialMaxPooling
   ReLU = cudnn.ReLU
   SpatialSoftMax = cudnn.SpatialSoftMax
else
   assert(false, 'Unknown backend type')
end


local net = nn.Sequential()
if opt.network == 'small' then
   print('==> init a small overfeat network')
   net:add(SpatialConvolution(3, 96, 11, 11, 4, 4))
   net:add(ReLU(opt.inplace))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolution(96, 256, 5, 5, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolution(1024, 3072, 6, 6, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(3072, 4096, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(4096, 1000, 1, 1, 1, 1))
   if not opt.spatial then net:add(nn.View(1000)) end
   net:add(SpatialSoftMax())
   print(net)

   -- init file pointer
   print('==> overwrite network parameters with pre-trained weigts')
   ParamBank:init("net_weight_0")
   ParamBank:read(        0, {96,3,11,11},    net:get(1).weight)
   ParamBank:read(    34848, {96},            net:get(1).bias)
   ParamBank:read(    34944, {256,96,5,5},    net:get(4).weight)
   ParamBank:read(   649344, {256},           net:get(4).bias)
   ParamBank:read(   649600, {512,256,3,3},   net:get(7).weight)
   ParamBank:read(  1829248, {512},           net:get(7).bias)
   ParamBank:read(  1829760, {1024,512,3,3},  net:get(9).weight)
   ParamBank:read(  6548352, {1024},          net:get(9).bias)
   ParamBank:read(  6549376, {1024,1024,3,3}, net:get(11).weight)
   ParamBank:read( 15986560, {1024},          net:get(11).bias)
   ParamBank:read( 15987584, {3072,1024,6,6}, net:get(14).weight)
   ParamBank:read(129233792, {3072},          net:get(14).bias)
   ParamBank:read(129236864, {4096,3072,1,1}, net:get(16).weight)
   ParamBank:read(141819776, {4096},          net:get(16).bias)
   ParamBank:read(141823872, {1000,4096,1,1}, net:get(18).weight)
   ParamBank:read(145919872, {1000},          net:get(18).bias)

elseif opt.network == 'big' then
   print('==> init a big overfeat network')
   net:add(SpatialConvolution(3, 96, 7, 7, 2, 2))
   net:add(ReLU(opt.inplace))
   net:add(SpatialMaxPooling(3, 3, 3, 3))
   net:add(SpatialConvolution(96, 256, 7, 7, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialMaxPooling(3, 3, 3, 3))
   net:add(SpatialConvolution(1024, 4096, 5, 5, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(4096, 4096, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(4096, 1000, 1, 1, 1, 1))
   if not opt.spatial then net:add(nn.View(1000)) end
   net:add(SpatialSoftMax())
   print(net)

   -- init file pointer
   print('==> overwrite network parameters with pre-trained weigts')
   ParamBank:init("net_weight_1")
   ParamBank:read(        0, {96,3,7,7},      net:get(1).weight)
   ParamBank:read(    14112, {96},            net:get(1).bias)
   ParamBank:read(    14208, {256,96,7,7},    net:get(4).weight)
   ParamBank:read(  1218432, {256},           net:get(4).bias)
   ParamBank:read(  1218688, {512,256,3,3},   net:get(7).weight)
   ParamBank:read(  2398336, {512},           net:get(7).bias)
   ParamBank:read(  2398848, {512,512,3,3},   net:get(9).weight)
   ParamBank:read(  4758144, {512},           net:get(9).bias)
   ParamBank:read(  4758656, {1024,512,3,3},  net:get(11).weight)
   ParamBank:read(  9477248, {1024},          net:get(11).bias)
   ParamBank:read(  9478272, {1024,1024,3,3}, net:get(13).weight)
   ParamBank:read( 18915456, {1024},          net:get(13).bias)
   ParamBank:read( 18916480, {4096,1024,5,5}, net:get(16).weight)
   ParamBank:read(123774080, {4096},          net:get(16).bias)
   ParamBank:read(123778176, {4096,4096,1,1}, net:get(18).weight)
   ParamBank:read(140555392, {4096},          net:get(18).bias)
   ParamBank:read(140559488, {1000,4096,1,1}, net:get(20).weight)
   ParamBank:read(144655488, {1000},          net:get(20).bias)
else
   assert(false, 'Unknown network type')
end
-- close file pointer
ParamBank:close()


-- load and preprocess image
print('==> prepare an input image')
local img = image.load(opt.img):mul(255)

-- use image larger than the eye size in spatial mode
if not opt.spatial then
   local dim = (opt.network == 'small') and 231 or 221
   local img_scale = image.scale(img, '^'..dim)
   local h = math.ceil((img_scale:size(2) - dim)/2)
   local w = math.ceil((img_scale:size(3) - dim)/2)
   img = image.crop(img_scale, w, h, w + dim, h + dim):floor()
end


-- memcpy from system RAM to GPU RAM if cuda enabled
if opt.backend == 'cunn' or opt.backend == 'cudnn' then
   net:cuda()
   img = img:cuda()
end


-- save bare network (before its buffer filled with temp results)
print('==> save model to:', opt.save)
torch.save(opt.save, net)


-- feedforward network
print('==> feed the input image')
timer = torch.Timer()
img:add(-118.380948):div(61.896913)
local out = net:forward(img)


-- find output class name in non-spatial mode
if not opt.spatial then
   local prob, idx = torch.max(out, 1)
   print(label[idx:squeeze()], prob:squeeze())
end
print('Time elapsed: ' .. timer:time().real .. ' seconds')
