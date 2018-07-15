function im = drawbbox(im, bbox, c, color)
sizeI = size(im);
sizeI = sizeI(1:2);

range = max(1,bbox(3)-c):min(sizeI(2),bbox(3)+c);
for i=bbox(1):bbox(2)
    for j=range
        im(i,j,:) = color;
    end
end

range = max(1,bbox(4)-c):min(sizeI(2),bbox(4)+c);
for i=bbox(1):bbox(2)
    for j=range
        im(i,j,:) = color;
    end
end

range = max(1,bbox(1)-c):min(sizeI(1),bbox(1)+c);
for i=range
    for j=bbox(3):bbox(4)
        im(i,j,:) = color;
    end
end

range = max(1,bbox(2)-c):min(sizeI(1),bbox(2)+c);
for i=range
    for j=bbox(3):bbox(4)
        im(i,j,:) = color;
    end
end

end

