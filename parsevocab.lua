require 'torch'

--get vocab
vocab = torch.load('vocab.asc')

cuArray = torch.eye(83)

function getOneHotChar(c)
    local index = vocab[c]
    local oneHotChar = cuArray[index]
    return oneHotChar
end

function getOneHotStr(s)
    local oneHotStr = nil
    for c in s:gmatch"." do
        if not oneHotStr then
            oneHotStr = getOneHotChar(c)
        else
            oneHotStr = torch.cat(oneHotStr, getOneHotChar(c), 2)
        end
    end
    return oneHotStr:clone()
end

function getOneHotStrs(strs)

-- will be given as an array of strs to be converted into one 
-- hot array of arrays

    maxCharLen = 0

    for i = 1, #strs do
        charLen = #strs[i]
        if charLen > maxCharLen then
            maxCharLen = charLen
        end 
    end
    
    --allOneHot = torch.zeros(83, maxCharLen, #strs)
    allOneHot = torch.zeros(#strs, maxCharLen, 83)
    
    for i = 1, #strs do
        strLen = #(strs[i])
        charRemain = maxCharLen - strLen
        oneHot = getOneHotStr(strs[i])
        if charRemain > 0 then
            zeroOneHotVectors = torch.zeros(83, charRemain)
            finalOneHot = torch.cat(oneHot, zeroOneHotVectors,2)
            allOneHot[{{i},{},{}}] = finalOneHot:t()
        else
            allOneHot[{{i},{},{}}] = oneHot:t()
        end
    end 

    return allOneHot
end
