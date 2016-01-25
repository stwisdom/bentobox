function y=pagefun_mtimes(x1,x2)

%     fid=fopen('log_pagefun_mtimes','a');
    ftic=tic;

    global fxns;

    x1sz=size(x1);
    x2sz=size(x2);
    
    if (x1sz(1)<16 && x1sz(2)<16) || (x1sz(2)==2)

%         x2=reshape(x2,[1,x2sz]);
        x2=permute(x2,[2,1,3:length(x2sz)]);
        
        if length(x1sz)>length(x2sz)
            x2sz=[x2sz, ones(1,length(x1sz)-length(x2sz))];
        elseif length(x1sz)<length(x2sz)
            x1sz=[x1sz, ones(1,length(x2sz)-length(x1sz))];
        end
        
        ysz=[x1sz(1),x2sz(2),max([x1sz(3:end); x2sz(3:end)],[],1)];
        y=fxns.gpuArray(zeros(ysz));
        for ii=1:x1sz(1)
            colon_rep=repmat(',:',[1,length(ysz)-1]);
            colon_rep_p1=[colon_rep ',:'];
            eval(['y(ii' colon_rep ')=reshape( sum( bsxfun(@times,x1(ii' colon_rep_p1 '),x2) ,2) ,[1,ysz(2:end)]);']);
        end

    else
        
%         L1=length(x1sz);
%         [~,imin1]=min(x1sz([1,3:L1]));
%         [~,imax1]=max(x1sz([1,3:L1]));
%         i1=setdiff(1:L1,[imin1,imax1,2]);
%         L2=length(x2sz);
%         [~,imin2]=min(x2sz(   2:L2));
%         [~,imax2]=max(x2sz(   2:L2));
%         i2=setdiff(1:L2,[imin2,imax2,1]);
%         
%         x1perm=[imax1,2,imin1,i1];
%         x1=permute(x1,x1perm);
%         x2perm=[1,imax2,imin2,i2];
%         x1=permute(x2,x2perm);
        
        y=pagefun(@mtimes,x1,x2);
        
    end
    
    ftoc=toc(ftic);
%     fprintf(fid,'x1 is ');
%     for ii=1:(length(x1sz)-1)
%         fprintf(fid,'%dx',x1sz(ii));
%     end
%     fprintf(fid,'%d, x2 is ',x1sz(end));
%     for ii=1:(length(x2sz)-1)
%         fprintf(fid,'%dx',x2sz(ii));
%     end
%     fprintf(fid,'%d, elapsed time is %.6f seconds.\n',x2sz(end),ftoc);
%     fclose(fid);
    