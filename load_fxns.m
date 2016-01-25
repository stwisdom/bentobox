function fxns=load_fxns(precFxn,flag_use_cpu)

    if ~exist('precFxn','var') || isempty(precFxn)
        precFxn=@(x)double(x);
    else
        if ~isa(precFxn,'function_handle')
            if precFxn
                precFxn=@(x)single(x);
            else
                precFxn=@(x)double(x);
            end
        end
    end
    
    if ~exist('flag_use_cpu','var') || isempty(flag_use_cpu)
        flag_use_cpu=0;
    end

    if gpuDeviceCount>0 && ~flag_use_cpu
        % GPU is present if we make it past the gpuDevice call,
        % so define GPU functions:
        disp('GPU is present, GPU will be used');
        fxns.gpuArray=@(x)gpuArray(precFxn(x));
        fxns.gather=@(x)double(gather(x));
        fxns.pagefun_marg=@(x,p,d)pagefun_marg_v2(fxns.gpuArray(x),fxns.gpuArray(p),d);
%         fxns.pagefun_marg=@(x,p,d)pagefun_marg(fxns.gpuArray(x),fxns.gpuArray(p),d);
%         fxns.pagefun_mtimes=@(x1,x2)pagefun(@mtimes,x1,x2);
        fxns.pagefun_mtimes=@(x1,x2)pagefun_mtimes(x1,x2); %uses bsxfun and sum if first dimension is small
        matlab_version = version;
        if ~isempty(strfind(matlab_version,'R2015a'))
            %fxns.pagefun_mldivide=@(A,x)pagefun(@mldivide,A,x);
            fxns.pagefun_mldivide=@(A,x)pagefun_mldivide_solve_lin_bkup(A,x);
        else
            fxns.pagefun_mldivide=@(A,x)fxns.gpuArray(solve_lin(fxns.gather(A),fxns.gather(x)));
        end
        fxns.pagefun_ctranspose=@(x)pagefun(@ctranspose,x);
    else
	if gpuDeviceCount>0
		disp('GPU is present, but CPU is specified; CPU will be used.');
	else
		disp('No GPU is present; CPU will be used.');
	end
        % define CPU functions:
        fxns.gpuArray=@(x)(precFxn(x));
        fxns.gather=@(x)double(gather(x));
        fxns.pagefun_marg=@(x,p,d)mtimes_marg(x,p,d);
        fxns.pagefun_mtimes=@(x1,x2)mtimesx(x1,x2);
        fxns.pagefun_mldivide=@(A,x)precFxn(solve_lin(double(A),double(x)));
        fxns.pagefun_ctranspose=@(x)cpu_pagefun_ctranspose(x);
    end
    
