function xm=mtimes_marg(x,p,dm)

    xsz=size(x);
    psz=size(p);
    
    if length(xsz)~=length(psz)
        
        xsz_gt_psz = length(xsz)>length(psz);
        
        if xsz_gt_psz
            psz=[psz ones(1,length(xsz)-length(psz))];
        else
            xsz=[xsz ones(1,length(psz)-length(xsz))];
        end
        
    end
    
    xr=reshape(x,[1,xsz]);
    pr=reshape(p,[1,psz]);
    
    xperm=[dm+1,1,setdiff(2:(1+length(xsz)),dm+1)];
    pperm=[1,dm+1,setdiff(2:(1+length(psz)),dm+1)];
    
    xm=mtimesx(permute(pr,pperm),permute(xr,xperm));
    
    xm=ipermute(xm,pperm);
    xmsz=size(xm);
    xm=reshape(xm,xmsz(2:end));
    

end
