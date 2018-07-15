function NTV = tvNormal(N,l)
    speck = zeros(3); speck(2,2) = 1;
    NTV = N;
    for c=1:3,
        tvOut = deconvtv(N(:,:,c),speck,l);
        NTV(:,:,c) = tvOut.f;
    end 
    NTV = bsxfun(@rdivide,NTV,sum(NTV.^2,3).^0.5+eps);
end
