function [gradApproxRe,gradApproxIm,itest]=gradientCheck(var_indep,fxn_update_dep,update_params,del,eps,testProportion,flag_nonneg)

    if ~exist('del','var') || isempty(del)
        del=1e-6;
    end
    
    if ~exist('eps','var') || isempty(eps)
        eps=1e-6;
    end
    
    if ~exist('testProportion','var')
        testProportion=1;
    end
    
    if ~exist('flag_nonneg','var')
        flag_nonneg=0;
    end

    numel_indep=numel(var_indep);
    
    if testProportion<1 && testProportion>0
        % restrict to a random subset of independent variables for faster
        % gradient checks:
        itest=randperm(numel_indep,floor(testProportion*numel_indep));
        numel_indep=length(itest);
    elseif testProportion==1
        itest=1:numel_indep;
    else
        error('testProportion must be positive and less than or equal to 1!');
    end
    
    flag_iscomplex_indep=1;
    if isreal(var_indep)
        flag_iscomplex_indep=0;
    end
    
    var_dep = fxn_update_dep(var_indep,update_params);
    
    if numel(var_dep)==1
        gradApproxRe=zeros(numel_indep,1);
        if flag_iscomplex_indep
            gradApproxIm=zeros(numel_indep,1);
        else
            gradApproxIm=[];
        end
    else
        gradApproxRe=cell(numel_indep,1);
        if flag_iscomplex_indep
            gradApproxIm=cell(numel_indep,1);
        else
            gradApproxIm=[];
        end
    end
    
    ttic=tic;
    fprintf('Checking %d total independent variables...\n',numel_indep);
    for ii=1:length(itest)
        
        if mod(ii,floor(numel_indep/10))==0
            fprintf('Checked %d independent variables of %d total\n',ii,numel_indep);
            toc(ttic);
            ttic=tic;
        end
        
        %%% Step 1: perturb one element of independent variable a bit:
        
        % add a bit to real part:
        var_indep_plusReal = var_indep;
        var_indep_plusReal(itest(ii)) = var_indep_plusReal(itest(ii)) + del;
        
        % sub a bit from real part:
        var_indep_minusReal = var_indep;
        var_indep_minusReal(itest(ii)) = var_indep_minusReal(itest(ii)) - del;
        
        if flag_nonneg
            if var_indep_minusReal(itest(ii))<0
                 var_indep_minusReal(itest(ii))=var_indep(itest(ii));
                 var_indep_plusReal(itest(ii))=var_indep_plusReal(itest(ii)) + del;
            end
        end
        
        if flag_iscomplex_indep
            
            % add a bit to imag part:
            var_indep_plusImag = var_indep;
            var_indep_plusImag(itest(ii)) = var_indep_plusImag(itest(ii)) + 1i*eps;
        
            % sub a bit from imag part:
            var_indep_minusImag = var_indep;
            var_indep_minusImag(itest(ii)) = var_indep_minusImag(itest(ii)) - 1i*eps;
            
        end
        
        %%% Step 2: compute values of dependent variable from perturbed
        %%% independent variable:
        
        var_dep_plusReal = fxn_update_dep(var_indep_plusReal,update_params);
        var_dep_minusReal = fxn_update_dep(var_indep_minusReal,update_params);
        if flag_iscomplex_indep
            var_dep_plusImag = fxn_update_dep(var_indep_plusImag,update_params);
            var_dep_minusImag = fxn_update_dep(var_indep_minusImag,update_params);
        end
        
        %%% Step 3: approximate the gradient

        gradApproxReCur = (var_dep_plusReal - var_dep_minusReal)./(2*del);
        
        if flag_iscomplex_indep
            gradApproxImCur = 1i.*(var_dep_plusImag - var_dep_minusImag)./(2*eps);
        end
        
        if numel(gradApproxReCur)==1
            % one-dimensional dep. var. (e.g., a cost function); use an
            % array to return:
            gradApproxRe(ii) = gather(gradApproxReCur);
            if flag_iscomplex_indep
                gradApproxIm(ii) = gather(gradApproxImCur);
            end
        else
            % multidimensional dep. var.; use a cell array to return
            gradApproxRe{ii} = gather(gradApproxReCur);
            if flag_iscomplex_indep
                gradApproxIm{ii} = gather(gradApproxImCur);
            end
        end
        
    end
    
    if testProportion==1
        gradApproxRe = reshape(gradApproxRe,size(var_indep));
        if flag_iscomplex_indep
            gradApproxIm = reshape(gradApproxIm,size(var_indep));
        end
    end

end

