require 'nn'
require 'torch'
require 'nngraph'
require 'windowMat2_nocuda'
require 'yHatMat2_nocuda'
--require 'windowMat2'
--require 'yHatMat2'
require 'parsevocab'
require 'distributions'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'

-- change model name here ---
model = torch.load('alexnetd.t7')

-- change test string here --
cu = getOneHotStrs({[1]="jimmy"})

--print(cu)
iteration = 1000
x_val = {[1]=0}
y_val = {[1]=0}
e_val = {[1]=0}
w = cu[{{},{1},{}}]:squeeze(2)
--print(w)
lstm_c_h1 = torch.zeros(1, 400)
lstm_h_h1 = torch.zeros(1, 400)
lstm_c_h2 = torch.zeros(1, 400)
lstm_h_h2 = torch.zeros(1, 400)
lstm_c_h3 = torch.zeros(1, 400)
lstm_h_h3 = torch.zeros(1, 400)
kappaprev = torch.zeros(1, 10)

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
    
    x_3 = torch.Tensor(1)
    x_3 = (x_3:bernoulli(e_t:squeeze())):squeeze()
    
    choice = {}
    
    for k=1,50 do
       table.insert(choice, distributions.cat.rnd(pi_t:squeeze(1)):squeeze()) 
    end
    chosen_pi = torch.multinomial(pi_t, 1):squeeze()
    --print(pi_t)
    --chosen_pi = distributions.cat.rnd(pi_t:squeeze(1)):squeeze()
    
    randChoice = torch.random(50)
    
    max = 0
    for i=1,20 do
        cur = pi_t[{{},{i}}]:squeeze()
        --print(cur)
        if cur > max then
           max = cur 
            index = i
        end
    end
    --chosen_pi = choice[randChoice]
    --print(chosen_pi)
    --chosen_pi=index
    --print(chosen_pi)
    
    curstd = torch.Tensor({{sigma_1_t[{{},{chosen_pi}}]:squeeze(), sigma_2_t[{{},{chosen_pi}}]:squeeze()}})
    curcor = torch.Tensor({{1, rho_t[{{},{chosen_pi}}]:squeeze()}})
    curcovmat = makecov(curstd, curcor)
    curmean = torch.Tensor({{mu_1_t[{{},{chosen_pi}}]:squeeze(), mu_2_t[{{},{chosen_pi}}]:squeeze()}})
    sample = distributions.mvn.rnd(curmean, curcovmat)
    x_1 = sample[1]
    x_2 = sample[2]
    
    table.insert(x_val, x_1)
    table.insert(y_val, x_2)
    table.insert(e_val, x_3)
    --return x_1, x_2, x_3
end

for t=1, 1000 do
    x_in = torch.Tensor({{x_val[t], y_val[t], e_val[t]}})
    
-- model 
    output_y, kappaprev, w, phi, lstm_c_h1, lstm_h_h1,
    lstm_c_h2, lstm_h_h2, lstm_c_h3, lstm_h_h3
    = unpack(model.rnn_core:forward({x_in, cu, 
    kappaprev, w, lstm_c_h1, lstm_h_h1,
    lstm_c_h2, lstm_h_h2, lstm_c_h3, lstm_h_h3}))    
    
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

mean_x = 8.1852355051148
mean_y = 0.1242846623983
std_x = 41.557732170832
std_y = 37.03681538566

for t=1,1000 do
    x_val[t] = (x_val[t]*std_x) + mean_x
    y_val[t] = (y_val[t]*std_y) + mean_y
end

r = torch.zeros(2, 200)

rs = {}
new = true
count = 0
i = 0
oldx = 0
oldy = 0
for j=1,1000 do
    i = i + 1
    if new then
        count = count + 1
        table.insert(rs, torch.zeros(2, 1000))
        i = 1
        new = false
    end
    
    if e_val[j] == 1 or j == 1000 then
        new = true
        rs[count] = rs[count][{{},{1,i}}]
    end
    
    newx = oldx + x_val[j]
    newy = oldy - y_val[j] 
    oldx = newx
    oldy = newy
    rs[count][{{1},{i}}] = newx
    rs[count][{{2},{i}}] = newy
    
end

plotter = {}

for i=1,#rs do
    table.insert(plotter, {"test", rs[i][1], rs[i][2], "-"})
end

gnuplot.plot(plotter)
gnuplot.axis({0,5000,0,5000})
