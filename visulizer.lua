require 'gnuplot'
require 'nn'
require 'cunn'
require 'torch'
require 'nngraph'
require 'windowMat2'
require 'yHatMat2'
require 'parsevocab'
require 'distributions'
require 'mixtureCriterionMat'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'

local cmd = torch.CmdLine() 
cmd:option('-string' , '', 'input string')
cmd:option('-batchSize', 1, 'batch size from trianing set') 
cmd:option('-sampling', false, 'sample the mixture output or using argmax')
cmd:option('-biassampling', 0., 'biased sampling the mixture output or using argmax')
cmd:option('-showphi', false, 'show matrix plot of phi')
cmd:option('-showgt', false, 'show ground truth from training set')
cmd:option('-feedgtinput', false, 'feed ground truth input sequence from training set')
cmd:option('-modelfile', 'alexnet.t7', 'load model file')

opt = cmd:parse(arg)
-- change model name here ---
model = torch.load(opt.modelfile)
model.rnn_core:double()
--params, grad_params = model.rnn_core:getParameters() 
--params:uniform(-0.08, 0.08)

-- change test string here --
if opt.string == '' then
    dataFile = torch.DiskFile('data_norm_mean_toy.asc', 'r')
    handwritingdata = dataFile:readObject()
    dataSize = #handwritingdata
    count = 1
    require 'getBatch'
    maxLen, strs, inputMat, cuMat, ymaskMat, wmaskMat, cmaskMat, elementCount,count = getBatch(count, handwritingdata, opt.batchSize)

    cu = cuMat
    opt.string = strs[1]
    --maxLen = 300
else
    cu = getOneHotStrs({[1]=opt.string})
    maxLen = 500
end

-- cu = getOneHotStrs({[1]="from nomnating any more Labour"})
-- cu = getOneHotStrs({[1]="A MOVE to stop Mr . Gaitskell"})
-- cu = getOneHotStrs({[1]="jimmy"})

phi_mat = torch.zeros(cu:size(2), maxLen)

--print(cu)
iteration = maxLen
x_val = {[1]=0}
y_val = {[1]=0}
e_val = {[1]=0}
w = cu[{{},{1},{}}]:squeeze(2)
--print(w)
lstm_c_h1 = torch.zeros(opt.batchSize, 400)
lstm_h_h1 = torch.zeros(opt.batchSize, 400)
lstm_c_h2 = torch.zeros(opt.batchSize, 400)
lstm_h_h2 = torch.zeros(opt.batchSize, 400)
lstm_c_h3 = torch.zeros(opt.batchSize, 400)
lstm_h_h3 = torch.zeros(opt.batchSize, 400)
kappaprev = torch.zeros(opt.batchSize, 10)

function makecov(std, rho)
    covmat = torch.Tensor(2,2)
    covmat[{{1},{1}}] = torch.pow(std[{{1},{1}}], 2)
    covmat[{{1},{2}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{1}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{2}}] = torch.pow(std[{{1},{2}}], 2)
    return covmat
end


function getX(input)
    
    e_t = input[{{},{1}}]
    pi_t = input[{{},{2,21}}]
    mu_1_t = input[{{},{22,41}}]
    mu_2_t = input[{{},{42,61}}]
    sigma_1_t = input[{{},{62,81}}]
    sigma_2_t = input[{{},{82,101}}]
    rho_t = input[{{},{102,121}}]
    
    -- decide end of stroke
    --x_3 = torch.Tensor(1)
    if opt.sampling == true then
        x_3 = (torch.bernoulli(e_t:squeeze()))
    else
        -- Threshold for end of stroke
        if e_t:squeeze() > 0.5 then
            x_3 = 1
        else
            x_3 = 0
        end
    end 

    -- decide mixture
    -- choice = {}
    
    --for k=1,50 do
    --   table.insert(choice, distributions.cat.rnd(pi_t:squeeze(1)):squeeze()) 
    --end
    --randChoice = torch.random(50)
    --chosen_pi = choice[randChoice]

    if opt.sampling == true then
        if opt.biassampling > 0. then
            pi_t = pi_t * opt.biassampling
        end
        chosen_pi = torch.multinomial(pi_t, 1):squeeze()
    --print(pi_t)
    --chosen_pi = distributions.cat.rnd(pi_t:squeeze(1)):squeeze()
    else 
        max = 0
        for i=1, pi_t:size(2) do
            cur = pi_t[{{},{i}}]:squeeze()
            --print(cur)
            if cur > max then
               max = cur 
                index = i
            end
        end
        chosen_pi=index
    end
    
    if opt.sampling == true then 
        if opt.biassampling > 0. then
            sigma_1_t:log():add(-opt.biassampling):exp()
            sigma_2_t:log():add(-opt.biassampling):exp()
        end
        curstd = torch.Tensor({{sigma_1_t[{{},{chosen_pi}}]:squeeze(), sigma_2_t[{{},{chosen_pi}}]:squeeze()}})
        curcor = torch.Tensor({{1, rho_t[{{},{chosen_pi}}]:squeeze()}})
        curcovmat = makecov(curstd, curcor)
        curmean = torch.Tensor({{mu_1_t[{{},{chosen_pi}}]:squeeze(), mu_2_t[{{},{chosen_pi}}]:squeeze()}})
        sample = distributions.mvn.rnd(curmean, curcovmat)

        x_1 = sample[1]
        x_2 = sample[2]
    else 
        -- choose mean
        x_1 = mu_1_t[{{},{chosen_pi}}]:squeeze()
        x_2 = mu_2_t[{{},{chosen_pi}}]:squeeze()
    end
    
    table.insert(x_val, x_1)
    table.insert(y_val, x_2)
    table.insert(e_val, x_3)
    return x_1, x_2, x_3
end

for t=1, maxLen do
    if opt.feedgtinput == true then
        x_in = inputMat[{{},{},{t}}]:squeeze(3)
    else
    x_in = torch.Tensor({{x_val[t], y_val[t], e_val[t]}})
    end
-- model 
    output_y, kappaprev, w, phi, lstm_c_h1, lstm_h_h1,
    lstm_c_h2, lstm_h_h2, lstm_c_h3, lstm_h_h3
    = unpack(model.rnn_core:forward({x_in, cu, 
    kappaprev, w, lstm_c_h1, lstm_h_h1,
    lstm_c_h2, lstm_h_h2, lstm_c_h3, lstm_h_h3}))
    
    --print(x_in[{{},{3}}])
    --print(e_t)

    phi_mat[{{},{t}}] = phi
    --print(t)
    --print(phi:squeeze()) 
    --print(phi:squeeze(1))
    getX(output_y)
    --sum_x_1 = 0
    --sum_x_2 = 0
    --sum_x_3 = 0
    
    --for i = 1, 10 do
      --  x_1, x_2, x_3 = getX(output_y)
       -- sum_x_1 = x_1 + sum_x_1
       -- sum_x_2 = x_2 + sum_x_2
       -- sum_x_3 = x_3 + sum_x_3
    --end
    
    --avg_x_1 = sum_x_1/10
    --print(avg_x_1)
    --avg_x_2 = sum_x_2/10
    --avg_x_3 = torch.round(sum_x_3/10)
    
    --table.insert(x_val, avg_x_1)
    --table.insert(y_val, avg_x_2)
    --table.insert(e_val, avg_x_3)
end


-- visualize x_val, y_val
mean_x = 8.1852355051148
mean_y = 0.1242846623983
std_x = 41.557732170832
std_y = 37.03681538566



function convert2plot(x_vec, y_vec, e_vec) 
    local rs = {}
    local new = true
    local count = 0
    local i = 0
    local oldx = 0
    local oldy = 0
    for j=1, maxLen do
        x_vec[j] = (x_vec[j]*std_x) + mean_x
        y_vec[j] = (y_vec[j]*std_y) + mean_y

        i = i + 1
        if new then
            count = count + 1
            table.insert(rs, torch.zeros(2, 1000))
            i = 1
            new = false
        end
        
        if e_vec[j] == 1 or j == 1000 then
            new = true
            rs[count] = rs[count][{{},{1,i}}]
        end
        
        local newx = oldx + x_vec[j]
        local newy = oldy - y_vec[j] 
        oldx = newx
        oldy = newy
        rs[count][{{1},{i}}] = newx
        rs[count][{{2},{i}}] = newy
    end
    return rs
end


ss = convert2plot(x_val, y_val, e_val)
plotter = {}
for i=1,#ss do
    table.insert(plotter, {"test", ss[i][1], ss[i][2], "-"})
end
gnuplot.figure(1)
gnuplot.plot(plotter)
gnuplot.axis({0,5000,-2500,2500})
gnuplot.title(opt.string)

gnuplot.pngfigure('sample_visual.png')
gnuplot.plot(plotter)
gnuplot.axis({0,5000,-2500,2500})
gnuplot.title(opt.string)
gnuplot.plotflush()

if opt.showphi == true then
gnuplot.pngfigure('phi_mat.png')
--gnuplot.figure(2)
gnuplot.imagesc(phi_mat:add(1.):log())
gnuplot.title(opt.string)
gnuplot.plotflush()
end

if opt.showgt == true then
gt = convert2plot(inputMat[{{},{1},{}}]:squeeze(), inputMat[{{},{2},{}}]:squeeze(), inputMat[{{},{3},{}}]:squeeze())
plotter_gt = {}
for i=1,#gt do
    table.insert(plotter_gt, {"gt", gt[i][1], gt[i][2], "-"})
end


gnuplot.pngfigure('gt_visual.png')
--gnuplot.figure(3)
gnuplot.plot(plotter_gt)
gnuplot.axis({0,5000,-2500,2500})
gnuplot.title(opt.string)
gnuplot.plotflush()
end


