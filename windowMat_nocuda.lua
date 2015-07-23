require 'nn'

local Window, Parent = torch.class('nn.Window', 'nn.Module')

function Window:setcu(cu)
    self.cu = cu
    self.cu_size = (#self.cu[{{1},{},{}}])[2]
    self.vocab_size = (#self.cu[{{1},{},{}}])[3]
end

function Window:setmask(mask)
    self.mask = mask
end

function Window:setKappaPrev(kappas_t_1)        
    self.kappas_t_1 = kappas_t_1
end

function Window:setGradKappaNext(d_kappas_t_plus_1)
    self.d_kappas_t_plus_1 = d_kappas_t_plus_1
end

function Window:updateOutput(input)
    input_cp = input:clone()
    
    hat_alphas_t = input_cp[{{},{1,10}}]
    hat_betas_t = input_cp[{{},{11,20}}]
    hat_kappas_t = input_cp[{{},{21,30}}]
    
    alphas_t = torch.exp(hat_alphas_t)
    betas_t = torch.exp(hat_betas_t)
    kappas_t = self.kappas_t_1 + torch.exp(hat_kappas_t)
    
    sampleSize = (#alphas_t)[1]
    
    u_vector = torch.linspace(1, self.cu_size, self.cu_size)
    
    u_expanded = u_vector:resize(1, 1, self.cu_size):expand(sampleSize, 10, self.cu_size)
    
    kappas_t_expanded = kappas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    betas_t_expanded = betas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    alphas_t_expanded = alphas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    calc = torch.pow(kappas_t_expanded - u_expanded, 2)
    
    calc:cmul(-betas_t_expanded)
    
    calc:exp()
    
    calc:cmul(alphas_t_expanded)
    
    phi_t = torch.sum(calc, 2)
    
    cu_resized = cu:clone()
    
    output = torch.bmm(phi_t, cu_resized):squeeze(2)
    
    self.output = output
    
    return self.output
end

function Window:updateGradInput(input, gradOutput)
    input_cp = input:clone()
    
    hat_alphas_t = input_cp[{{},{1,10}}]
    hat_betas_t = input_cp[{{},{11,20}}]
    hat_kappas_t = input_cp[{{},{21,30}}]
     
    alphas_t = torch.exp(hat_alphas_t)
    betas_t = torch.exp(hat_betas_t)
    kappas_t = self.kappas_t_1 + torch.exp(hat_kappas_t)
    
    sampleSize = (#alphas_t)[1]
    
    -- calculate epsilon(k,t,u)
    
    gradOutput_expanded = gradOutput:clone():resize(sampleSize, 1, self.vocab_size)
    :expand(sampleSize, self.cu_size, self.vocab_size)
    
    cu_resized = cu:clone()
    
    calc = torch.cmul(gradOutput_expanded, cu_resized)
    
    calc = calc:sum(3):squeeze(3)
    
    gradSum = calc:clone():resize(sampleSize, 1, self.cu_size)
    :expand(sampleSize, 10, self.cu_size)
    
    u_vector = torch.linspace(1, self.cu_size, self.cu_size)
    
    u_expanded = u_vector:resize(1, 1, self.cu_size):expand(sampleSize, 10, self.cu_size)
    
    kappas_t_expanded = kappas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    betas_t_expanded = betas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    alphas_t_expanded = alphas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    calc = torch.pow(kappas_t_expanded - u_expanded, 2)
    
    calc:cmul(-betas_t_expanded)
    
    calc:exp()
    
    calc:cmul(alphas_t_expanded)
    
    epsilon = torch.cmul(calc, gradSum)
    
    --compute dl_dalphas_hat 
    dl_dalphas_hat = torch.sum(epsilon, 3):squeeze(3)
    
    --compute dl_dbetas_hat
    dl_dbetas_hat = torch.pow(kappas_t_expanded - u_expanded, 2)
    dl_dbetas_hat:cmul(epsilon)
    dl_dbetas_hat = torch.sum(dl_dbetas_hat, 3):squeeze(3)
    dl_dbetas_hat:cmul(-betas_t)

    --compute dl_dkappas
    dl_dkappas = torch.cmul(epsilon, u_expanded - kappas_t_expanded)
    dl_dkappas = torch.sum(dl_dkappas, 3):squeeze(3)
    dl_dkappas:cmul(betas_t)
    dl_dkappas:mul(2)
    dl_dkappas:add(self.d_kappas_t_plus_1)
    
    --compute dl_dkappas_hat
    dl_dkappas_hat = torch.cmul(torch.exp(hat_kappas_t), dl_dkappas)
    
    self.gradInput = torch.cat(dl_dalphas_hat:double(), torch.cat(dl_dbetas_hat:double(), dl_dkappas_hat:double(), 2), 2)
    self.gradInput = self.gradInput:squeeze()
    self.gradInput = torch.cmul(self.gradInput, self.mask)
    self.gradInput:resize(sampleSize, 30)
    
    return self.gradInput
end
