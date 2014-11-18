require 'nn'
require 'image'
local ParamBank = require 'ParamBank'
local label     = require 'overfeat_label'

local cuda=true
require 'cunn'
require 'cudnn'

-- OverFeat input arguements
local network  = 'small' or 'big'
local filename = 'bee.jpg'

-- system parameters
local threads = 4
local offset  = 0

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)
print('==> #threads:', torch.getnumthreads())


local function nilling(module)
   module.gradBias   = nil
   if module.finput then module.finput = torch.Tensor():typeAs(module.finput) end
   module.gradWeight = nil
   module.output     = torch.Tensor():typeAs(module.output)
   module.fgradInput = nil
   module.gradInput  = nil
end

local function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end


net = nn.Sequential()
local m = net.modules
if network == 'small' then
   print('==> init a small overfeat network')
   net:add(cudnn.SpatialConvolution(3, 96, 11, 11, 4, 4))
   net:add(nn.Threshold(0.000001, 0.00000))
   net:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   net:add(cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1))
   net:add(nn.Threshold(0.000001, 0.00000))
   net:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   net:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
   net:add(nn.Threshold(0.000001, 0.00000))
   net:add(cudnn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
   net:add(nn.Threshold(0.000001, 0.00000))
   net:add(cudnn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
   net:add(nn.Threshold(0.000001, 0.00000))
   net:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   net:add(cudnn.SpatialConvolution(1024, 3072, 6, 6, 1, 1))
   net:add(nn.Threshold(0.000001, 0.00000))
   net:add(cudnn.SpatialConvolution(3072, 4096, 1, 1, 1, 1))
   net:add(nn.Threshold(0.000001, 0.00000))
   net:add(cudnn.SpatialConvolution(4096, 1000, 1, 1, 1, 1))
   net:add(cudnn.SpatialSoftMax())
   net = net:float()
   print(net)

   local mods = net:findModules('cudnn.SpatialConvolution')
   -- init file pointer
   print('==> overwrite network parameters with pre-trained weigts')
   ParamBank:init("net_weight_0")
   ParamBank:read(        0, {96,3,11,11},    mods[1].weight)
   ParamBank:read(    34848, {96},            mods[1].bias)
   ParamBank:read(    34944, {256,96,5,5},    mods[2].weight)
   ParamBank:read(   649344, {256},           mods[2].bias)
   ParamBank:read(   649600, {512,256,3,3},   mods[3].weight)
   ParamBank:read(  1829248, {512},           mods[3].bias)
   ParamBank:read(  1829760, {1024,512,3,3},  mods[4].weight)
   ParamBank:read(  6548352, {1024},          mods[4].bias)
   ParamBank:read(  6549376, {1024,1024,3,3}, mods[5].weight)
   ParamBank:read( 15986560, {1024},          mods[5].bias)
   ParamBank:read( 15987584, {3072,1024,6,6}, mods[6].weight)
   ParamBank:read(129233792, {3072},          mods[6].bias)
   ParamBank:read(129236864, {4096,3072,1,1}, mods[7].weight)
   ParamBank:read(141819776, {4096},          mods[7].bias)
   ParamBank:read(141823872, {1000,4096,1,1}, mods[8].weight)
   ParamBank:read(145919872, {1000},          mods[8].bias)

   torch.save('overfeat.net.small.cudnn.t7', net)
elseif network == 'big' then
   print('==> init a big overfeat network')
   net:add(cudnn.SpatialConvolution(3, 96, 7, 7, 2, 2))
   net:add(nn.Threshold(0, 0.000001))
   net:add(cudnn.SpatialMaxPooling(3, 3, 3, 3))
   net:add(cudnn.SpatialConvolution(96, 256, 7, 7, 1, 1))
   net:add(nn.Threshold(0, 0.000001))
   net:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   net:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
   net:add(nn.Threshold(0, 0.000001))
   net:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
   net:add(nn.Threshold(0, 0.000001))
   net:add(cudnn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
   net:add(nn.Threshold(0, 0.000001))
   net:add(cudnn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
   net:add(nn.Threshold(0, 0.000001))
   net:add(cudnn.SpatialMaxPooling(3, 3, 3, 3))
   net:add(cudnn.SpatialConvolution(1024, 4096, 5, 5, 1, 1))
   net:add(nn.Threshold(0, 0.000001))
   net:add(cudnn.SpatialConvolution(4096, 4096, 1, 1, 1, 1))
   net:add(nn.Threshold(0, 0.000001))
   net:add(cudnn.SpatialConvolution(4096, 1000, 1, 1, 1, 1))
   net:add(cudnn.SpatialSoftMax())
   net = net:float()
   print(net)

   local mods = net:findModules('cudnn.SpatialConvolution')
   -- init file pointer
   print('==> overwrite network parameters with pre-trained weigts')
   ParamBank:init("net_weight_1")
   ParamBank:read(        0, {96,3,7,7},      mods[1].weight)
   ParamBank:read(    14112, {96},            mods[1].bias)
   ParamBank:read(    14208, {256,96,7,7},    mods[2].weight)
   ParamBank:read(  1218432, {256},           mods[2].bias)
   ParamBank:read(  1218688, {512,256,3,3},   mods[3].weight)
   ParamBank:read(  2398336, {512},           mods[3].bias)
   ParamBank:read(  2398848, {512,512,3,3},   mods[4].weight)
   ParamBank:read(  4758144, {512},           mods[4].bias)
   ParamBank:read(  4758656, {1024,512,3,3},  mods[5].weight)
   ParamBank:read(  9477248, {1024},          mods[5].bias)
   ParamBank:read(  9478272, {1024,1024,3,3}, mods[6].weight)
   ParamBank:read( 18915456, {1024},          mods[6].bias)
   ParamBank:read( 18916480, {4096,1024,5,5}, mods[7].weight)
   ParamBank:read(123774080, {4096},          mods[7].bias)
   ParamBank:read(123778176, {4096,4096,1,1}, mods[8].weight)
   ParamBank:read(140555392, {4096},          mods[8].bias)
   ParamBank:read(140559488, {1000,4096,1,1}, mods[9].weight)
   ParamBank:read(144655488, {1000},          mods[9].bias)

   torch.save('overfeat.net.big.cudnn.t7', net)

end
-- close file pointer
ParamBank:close()

if cuda then net:cuda() end

-- load and preprocess image
print('==> prepare an input image')
local img = image.load(filename):mul(255)
-- feedforward network
print('==> feed the input image')
timer = torch.Timer()
img:add(-118.380948):div(61.896913)  -- fixed distn ~ N(118.380948, 61.896913^2)
img=img:cuda()
local out = net:forward(img):clone():float()
print(#out)
prob, idx = torch.max(out[1], 1)
print('Time elapsed: ' .. timer:time().real .. ' seconds')
