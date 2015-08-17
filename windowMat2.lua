require 'nn'

local Window, Parent = torch.class('nn.Window', 'nn.Module')

function Window:__init(nWindow)
    self.nWindow = nWindow
end

function Window:updateOutput(input)
    local input_h1, context, kappas_t_1, _ = unpack(input)

    self.cu = context
    self.cu_size = (#self.cu[{{1},{},{}}])[2]
    self.vocab_size = (#self.cu[{{1},{},{}}])[3]
    
    local input_cp = input_h1:clone()
    local slice_idx = 1 
    local hat_alphas_t = input_cp[{{},{slice_idx,slice_idx+self.nWindow-1}}]
    slice_idx = slice_idx+self.nWindow
    local hat_betas_t = input_cp[{{},{slice_idx,slice_idx+self.nWindow-1}}]
    slice_idx = slice_idx+self.nWindow
    local hat_kappas_t = input_cp[{{},{slice_idx,slice_idx+self.nWindow-1}}]
    
    local sampleSize = (#input_h1)[1]

    local alphas_t = hat_alphas_t:clone():exp():resize(sampleSize, self.nWindow, 1):expand(sampleSize, self.nWindow, self.cu_size)
    self.betas_t = hat_betas_t:clone():exp()
    local betas_t = self.betas_t:clone():resize(sampleSize, self.nWindow, 1):expand(sampleSize, self.nWindow, self.cu_size)
    local kappas_t = kappas_t_1:clone():add(hat_kappas_t:clone():exp())
    local kappas_t_expanded = kappas_t:clone():resize(sampleSize, self.nWindow, 1):expand(sampleSize, self.nWindow, self.cu_size)
    
    local u_vector = input_h1.torch.zeros(self.cu_size):copy(torch.linspace(1, self.cu_size, self.cu_size)):resize(1, 1, self.cu_size):expand(sampleSize, self.nWindow, self.cu_size)
    
    --- calculate the weighted window
    self.offset = u_vector:clone():add(-kappas_t_expanded)
    self.mixtureK = self.offset:clone():pow(2):cmul(-betas_t):exp():cmul(alphas_t)
    local phi_t = self.mixtureK:sum(2):squeeze(2)
    local cu_resized = self.cu:clone()
    --print(cu_resized)
    local output = phi_t:clone():resize(sampleSize, self.cu_size, 1):expand(sampleSize, self.cu_size, self.vocab_size):clone():cmul(self.cu):sum(2):squeeze(2)
    
    --local output = input_h1.torch.bmm(phi_t, cu_resized):squeeze(2) 
    self.output = {output, kappas_t, phi_t}
    
    return self.output
end

function Window:updateGradInput(input, gradOutput)
    local input_h1, context, kappas_t_1, mask = unpack(input)
    local grad_output, d_kappas_t_plus_1 = unpack(gradOutput)
    
    local input_cp = input_h1:clone()
    local slice_idx = 1 
    local hat_kappas_t = input_cp[{{},{slice_idx+self.nWindow*2,slice_idx+self.nWindow*3-1}}]
    
    local sampleSize = (#input_cp)[1]
   
    -- calculate epsilon(k,t,u)
    
    local gradOutput_expanded = grad_output:clone():resize(sampleSize, 1, self.vocab_size)
    :expand(sampleSize, self.cu_size, self.vocab_size)
    
    local epsilon = self.mixtureK:clone():cmul(gradOutput_expanded:clone():cmul(context):sum(3):resize(sampleSize, 1, self.cu_size):expand(sampleSize, self.nWindow, self.cu_size))
    
    --compute dl_dalphas_hat 
    local dl_dalphas_hat = epsilon:sum(3):squeeze(3)
    
    --compute dl_dbetas_hat
    local dl_dbetas_hat = self.betas_t:clone():mul(-1):cmul(epsilon:clone():cmul(self.offset:clone():pow(2)):sum(3):squeeze(3))
    
    --compute dl_dkappas
    --print(d_kappas_t_plus_1)
    local dl_dkappas = self.betas_t:clone():mul(2):cmul(epsilon:clone():cmul(self.offset):sum(3):squeeze(3)):add(d_kappas_t_plus_1)
    
    --compute dl_dkappas_hat
    local dl_dkappas_hat = hat_kappas_t:clone():exp():cmul(dl_dkappas)
 
    self.joinFunc = self.joinFunc or nn.JoinTable(2)
    if input_h1:type() == 'torch.CudaTensor' then
        self.joinFunc:cuda()
    end

    local grad_input = self.joinFunc:forward({dl_dalphas_hat,dl_dbetas_hat, dl_dkappas_hat})
    local grad_context = context:clone():zero()
   
    self.gradInput = {grad_input, grad_context, dl_dkappas}
    
    return self.gradInput
    
end
