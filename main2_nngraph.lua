require 'getBatch'
--params:uniform(-0.08, 0.08)
sampleSize = 4
numberOfPasses = 8

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
initstate_h1_c = torch.zeros(sampleSize, 400):cuda()
initstate_h1_h = initstate_h1_c:clone()
initstate_h2_c = initstate_h1_c:clone()
initstate_h2_h = initstate_h1_c:clone()
initstate_h3_c = initstate_h1_c:clone()
initstate_h3_h = initstate_h1_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
dfinalstate_h1_c = initstate_h1_c:clone()
dfinalstate_h1_h = initstate_h1_c:clone()
dfinalstate_h2_c = initstate_h1_c:clone()
dfinalstate_h2_h = initstate_h1_c:clone()
dfinalstate_h3_c = initstate_h1_c:clone()
dfinalstate_h3_h = initstate_h1_c:clone()
initkappa = torch.randn(sampleSize,10)
dinitkappa = torch.zeros(sampleSize,10)

count = 1

function getInitW(cuMat)
    return cuMat[{{},{1},{}}]:squeeze(2)
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    
    local loss = 0
    local elems = 0
    
    -- add for loop to increase mini-batch size
    for i=1, numberOfPasses do

        --------------------- get mini-batch -----------------------
        maxLen, strs, inputMat, cuMat, ymaskMat, wmaskMat, cmaskMat, elementCount, 
        count = getBatch(count, handwritingdata, sampleSize)
        ------------------------------------------------------------

        if maxLen > MAXLEN then
            maxLen = MAXLEN
        end

        -- initialize window to first char in all elements of the batch
        local w = {[0]=getInitW(cuMat:cuda())}

        local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
        local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
        local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
        local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
        local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
        local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
        
        local kappa_prev = {[0]=torch.zeros(sampleSize,10):cuda()}
        
        local output_h1_w = {}
        local input_h3_y = {}
        local output_h3_y = {}
        local output_y = {}
        
        -- forward
        
        print('forward')
        
        for t = 1, maxLen - 1 do
            local x_in = inputMat[{{},{},{t}}]:squeeze()
            local x_target = inputMat[{{},{},{t+1}}]:squeeze()
       
            -- model 
            output_y[t], kappa_prev[t], w[t], lstm_c_h1[t], lstm_h_h1[t],
            lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t]
        = unpack(clones.rnn_core[t]:forward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1]}))
       
            -- criterion 
            clones.criterion[t]:setmask(cmaskMat[{{},{},{t}}]:cuda())
            loss = clones.criterion[t]:forward(output_y[t], x_target:cuda()) + loss
            --print('inner loop ',loss)        
        end
        print('current pass ',loss)        
        elems = (elementCount - sampleSize) + elems
        
        -- backward
        
        print('backward')
        
        local dlstm_c_h1 = dfinalstate_h1_c
        local dlstm_h_h1 = dfinalstate_h1_h
        local dlstm_c_h2 = dfinalstate_h2_c
        local dlstm_h_h2 = dfinalstate_h2_h
        local dlstm_c_h3 = dfinalstate_h3_c
        local dlstm_h_h3 = dfinalstate_h3_h
        
        local dh1_w = torch.zeros(sampleSize, 83):cuda()
        local dkappa = torch.zeros(sampleSize, 10):cuda()
        
        for t = maxLen - 1, 1, -1 do
        
            local x_in = inputMat[{{},{},{t}}]:squeeze()
            local x_target = inputMat[{{},{},{t+1}}]:squeeze()
            
            -- criterion
            local grad_crit = clones.criterion[t]:backward(output_y[t], x_target:cuda())
	    grad_crit:clamp(-100,100)            

            -- model
            _x, _c, dkappa, dh1_w, dlstm_c_h1, dlstm_h_h1,
            dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3 = unpack(clones.rnn_core[t]:backward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1]},
                 {grad_crit, dkappa, dh1_w, dlstm_c_h1, dlstm_h_h1, 
                  dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3 }))

            -- clip gradients
            dh1_w:clamp(-10,10)
            dlstm_c_h1:clamp(-10,10)
            dlstm_h_h1:clamp(-10,10)
            dlstm_c_h2:clamp(-10,10)
            dlstm_h_h2:clamp(-10,10)
            dlstm_c_h3:clamp(-10,10)
            dlstm_h_h3:clamp(-10,10)
        end
    
        dh2_w = nil
        dh2_h1 = nil
        dh3_w = nil
        dh3_h2 = nil
        maxLen = nil
        strs = nil
        inputMat = nil 
        maskMat = nil
        cuMat = nil
        w = nil
        lstm_c_h1 = nil -- internal cell states of LSTM
        lstm_h_h1 = nil -- output values of LSTM
        lstm_c_h2 = nil -- internal cell states of LSTM
        lstm_h_h2 = nil -- output values of LSTM
        lstm_c_h3 = nil -- internal cell states of LSTM
        lstm_h_h3 = nil -- output values of LSTM
        dlstm_c_h1 = nil -- internal cell states of LSTM
        dlstm_h_h1 = nil -- internal cell states of LSTM
        dlstm_c_h2 = nil -- internal cell states of LSTM
        dlstm_h_h2 = nil -- internal cell states of LSTM
        dlstm_c_h3 = nil -- internal cell states of LSTM
        dlstm_h_h3 = nil -- internal cell states of LSTM
        dkappaNext = nil
        dh1_w_next = nil
        kappa_prev = nil
        output_h1_w = nil
        input_h3_y = nil
        output_h3_y = nil
        output_y = nil
        collectgarbage()
    end
    
    grad_params:div(numberOfPasses)
    
    -- clip gradient element-wise
    grad_params:clamp(-10, 10)
    
    
    return loss, grad_params
end

losses = {} -- TODO: local
local optim_state = {learningRate = 1e-4, alpha = 0.95, epsilon = 1e-4}
local iterations = 8000
for i = 1, iterations do
    local _, loss = optim.rmsprop(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    print('update param, loss:',loss[1])

    if i % 5 == 0 then
        torch.save("alexnet.t7", model)
        torch.save("losses.t7", losses)
    end
    if i % 5 == 0 then
        print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], grad_params:norm()))
    end
end
