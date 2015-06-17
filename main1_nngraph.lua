require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'parsevocab'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'
require 'windowMat2'
require 'yHatMat2'
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
--model.h1 = LSTMH1.lstm():cuda()
--model.h1_w = nn.Linear(400, 30):cuda()
--model.w = nn.Window():cuda()
--model.h2 = LSTMHN.lstm():cuda()
--model.h3 = LSTMHN.lstm():cuda()
--model.h3_y = nn.Linear(1200, 121):cuda()
--model.y = nn.YHat():cuda()
model.criterion = nn.MixtureCriterion():cuda()
model.criterion:setSizeAverage()


local input_xin = nn.Identity()()
local input_context = nn.Identity()()
local input_w_prev = nn.Identity()()
local input_lstm_h1_h = nn.Identity()()
local input_lstm_h1_c = nn.Identity()()
local input_lstm_h2_h = nn.Identity()()
local input_lstm_h2_c = nn.Identity()()
local input_lstm_h3_h = nn.Identity()()
local input_lstm_h3_c = nn.Identity()()
local input_prev_kappa = nn.Identity()()


local h1 = LSTMH1.lstm()({input_xin, input_w_prev, input_lstm_h1_c, input_lstm_h1_h})
local h1_c = nn.SelectTable(1)(h1)
local h1_h = nn.SelectTable(2)(h1)
local w_output = nn.Window()({nn.Linear(400,30)(h1_h), input_context, input_prev_kappa})
local w_vector = nn.SelectTable(1)(w_output)
local w_kappas_t = nn.SelectTable(2)(w_output)
local h2 = LSTMHN.lstm()({input_xin, w_vector, h1_h, input_lstm_h2_c, input_lstm_h2_h})
local h2_c = nn.SelectTable(1)(h2)
local h2_h = nn.SelectTable(2)(h2)
local h3 = LSTMHN.lstm()({input_xin, w_vector, h2_h, input_lstm_h3_c, input_lstm_h3_h})
local h3_c = nn.SelectTable(1)(h3)
local h3_h = nn.SelectTable(2)(h3)
local y = nn.YHat()(nn.Linear(1200, 121)(nn.JoinTable(2)({h1_h, h2_h, h3_h})))

model.rnn_core = nn.gModule({input_xin, input_context, input_prev_kappa, input_w_prev,  
                             input_lstm_h1_c, input_lstm_h1_h,
                             input_lstm_h2_c, input_lstm_h2_h,
                             input_lstm_h3_c, input_lstm_h3_h},
                            {y, w_kappas_t, w_vector, h1_c, h1_h, h2_c, h2_h,
                             h3_c, h3_h})

model.rnn_core:cuda()
params, grad_params = model.rnn_core:getParameters()

--params, grad_params = model_utils.combine_all_parameters(model.h1, model.h1_w, model.w, model.h2, model.h3, model.h3_y, model.y)
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
    clones[name] = model_utils.clone_many_times_fast(mod, 999, not mod.parameters)
end
