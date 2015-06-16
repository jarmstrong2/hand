require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'parsevocab'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'
require 'windowMat'
require 'yHatMat'
require 'mixtureCriterionMat'
local model_utils=require 'model_utils'
require 'cunn'

torch.manualSeed(123)

-- get dataset
dataFile = torch.DiskFile('data_norm_mean_toy.asc', 'r')
handwritingdata = dataFile:readObject()
dataSize = #handwritingdata

-- make model
model = {}
model.h1 = LSTMH1.lstm():cuda()
model.h1_w = nn.Linear(400, 30):cuda()
model.w = nn.Window():cuda()
model.h2 = LSTMHN.lstm():cuda()
model.h3 = LSTMHN.lstm():cuda()
model.h3_y = nn.Linear(1200, 121):cuda()
model.y = nn.YHat():cuda()
criterion = nn.MixtureCriterion():cuda()

params, grad_params = model_utils.combine_all_parameters(model.h1, model.h1_w, model.w, model.h2, model.h3, model.h3_y, model.y)
params:uniform(-0.08, 0.08)

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
initstate_h1_c = torch.zeros(1, 400):cuda()
initstate_h1_h = initstate_h1_c:clone()
initstate_h2_c = initstate_h1_c:clone()
initstate_h2_h = initstate_h1_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
dfinalstate_h1_c = initstate_h1_c:clone()
dfinalstate_h1_h = initstate_h1_c:clone()
dfinalstate_h2_c = initstate_h1_c:clone()
dfinalstate_h2_h = initstate_h1_c:clone()

-- make a bunch of clones, AFTER flattening, as that reallocates memory
clones = {} -- TODO: local
for name,mod in pairs(model) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(mod, 999, not mod.parameters)
end
