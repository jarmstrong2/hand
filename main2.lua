require 'getBatch'
--params:uniform(-0.08, 0.08)
sampleSize = 4
numberOfPasses = 32

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

        if maxLen > 1000 then
            maxLen = 1000
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
        
            -- h1
            lstm_c_h1[t], lstm_h_h1[t] = unpack(clones.h1[t]:forward({x_in:cuda(), 
            w[t-1]:cuda(), lstm_c_h1[t-1], lstm_h_h1[t-1]}))
        
            -- h1_w
            output_h1_w[t] = clones.h1_w[t]:forward(lstm_h_h1[t])
        
            -- w
            clones.w[t]:setcu(cuMat:cuda())
            clones.w[t]:setmask(wmaskMat[{{},{},{t}}]:cuda())
            clones.w[t]:setKappaPrev(kappa_prev[t-1]:cuda())
            kappa_prev[t] = kappa_prev[t-1] + torch.exp(output_h1_w[t][{{},{21,30}}])
            w[t] = clones.w[t]:forward(output_h1_w[t])
        
            -- h2
            lstm_c_h2[t], lstm_h_h2[t] = unpack(clones.h2[t]:forward({x_in:cuda(), 
            w[t], lstm_h_h1[t], lstm_c_h2[t-1], lstm_h_h2[t-1]}))
        
            -- h3
            lstm_c_h3[t], lstm_h_h3[t] = unpack(clones.h3[t]:forward({x_in:cuda(), 
            w[t], lstm_h_h2[t], lstm_c_h3[t-1], lstm_h_h3[t-1]}))
        
            -- h3_y
            input_h3_y[t] = torch.cat(lstm_h_h1[t]:float(), lstm_h_h2[t]:float(), 2)
            input_h3_y[t] = torch.cat(input_h3_y[t], lstm_h_h3[t]:float(), 2)
            output_h3_y[t] = clones.h3_y[t]:forward(input_h3_y[t]:cuda())
        
            -- y
            clones.y[t]:settarget(x_target:cuda())
            clones.y[t]:setmask(ymaskMat[{{},{},{t}}]:cuda())
            output_y[t] = clones.y[t]:forward(output_h3_y[t])
        
            criterion:setmask(cmaskMat[{{},{},{t}}]:cuda())
            loss = criterion:forward(output_y[t]:cuda(), x_target:cuda()) + loss
        end
        
        elems = (elementCount - sampleSize) + elems
        
        -- backward
        
        print('backward')
        
        local dlstm_c_h1 = {[(maxLen - 1)] = dfinalstate_h1_c} -- internal cell states of LSTM
        local dlstm_h_h1 = {[(maxLen - 1)] = dfinalstate_h1_h} -- internal cell states of LSTM
        local dlstm_c_h2 = {[(maxLen - 1)] = dfinalstate_h2_c} -- internal cell states of LSTM
        local dlstm_h_h2 = {[(maxLen - 1)] = dfinalstate_h2_h} -- internal cell states of LSTM
        local dlstm_c_h3 = {[(maxLen - 1)] = dfinalstate_h3_c} -- internal cell states of LSTM
        local dlstm_h_h3 = {[(maxLen - 1)] = dfinalstate_h3_h} -- internal cell states of LSTM
        
        local dh1_w_next = torch.zeros(sampleSize, 83):cuda()
        local dkappaNext = torch.zeros(sampleSize, 10):cuda()
        
        for t = maxLen - 1, 1, -1 do
        
            local x_in = inputMat[{{},{},{t}}]:squeeze()
            local x_target = inputMat[{{},{},{t+1}}]:squeeze()
            
            -- y
            local grad_y = clones.y[t]:backward(output_h3_y[t]:cuda())
        
            -- h3_y
            local grad_h3_y = clones.h3_y[t]:backward(input_h3_y[t]:cuda(), grad_y:cuda())
            grad_h3_y:clamp(-100,100)
            dlstm_h_h1_y = grad_h3_y[{{},{1,400}}]
            dlstm_h_h2_y = grad_h3_y[{{},{401,800}}]
            dlstm_h_h3_y = grad_h3_y[{{},{801,1200}}]
        
            -- h3
            _, dh3_w, dh3_h2, dlstm_c_h3[t - 1], dlstm_h_h3[t - 1] = unpack(clones.h3[t]:backward(
            {x_in:cuda(), w[t], lstm_h_h2[t], lstm_c_h3[t-1], lstm_h_h3[t-1]},
            {dlstm_c_h3[t], dlstm_h_h3[t] + dlstm_h_h3_y}
            ))
            dh3_w:clamp(-10,10)
            dh3_h2:clamp(-10,10)
            dlstm_c_h3[t - 1]:clamp(-10,10)
            dlstm_h_h3[t - 1]:clamp(-10,10)
            
            -- h2
            _, dh2_w, dh2_h1, dlstm_c_h2[t - 1], dlstm_h_h2[t - 1] = unpack(clones.h2[t]:backward(
            {x_in:cuda(), w[t], lstm_h_h1[t], lstm_c_h2[t-1], lstm_h_h2[t-1]},
            {dlstm_c_h2[t], dlstm_h_h2[t] + dlstm_h_h2_y + dh3_h2}
            ))
            dh2_w:clamp(-10,10)
            dh2_h1:clamp(-10,10)
            dlstm_c_h2[t - 1]:clamp(-10,10)
            dlstm_h_h2[t - 1]:clamp(-10,10)
            
            -- w
            clones.w[t]:setGradKappaNext(dkappaNext:cuda())
            local grad_w = clones.w[t]:backward(output_h1_w[t], dh1_w_next + dh2_w + dh3_w) 
            dkappaNext = grad_w[{{},{21,30}}]
        
            -- h1_w
            local grad_h1_w = clones.h1_w[t]:backward(lstm_h_h1[t], grad_w)
        
            -- h1
            _, dh1_w_next, dlstm_c_h1[t-1], dlstm_h_h1[t-1] = unpack(clones.h1[t]:backward(
            {x_in:cuda(), w[t-1]:cuda(), lstm_c_h1[t-1], lstm_h_h1[t-1]},
            {dlstm_c_h1[t], dlstm_c_h1[t] + grad_h1_w + dh2_h1 + dlstm_h_h1_y}
            ))
            dh1_w_next:clamp(-10,10)
            dlstm_c_h1[t - 1]:clamp(-10,10)
            dlstm_h_h1[t - 1]:clamp(-10,10)
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
    loss = loss/elems
    
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

    print(loss[1])

    if i % 5 == 0 then
        torch.save("alexnet.t7", model)
        torch.save("losses.t7", losses)
    end
    if i % 5 == 0 then
        print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], grad_params:norm()))
    end
end