function [f_vec,minLoss] = pathTrace(imageMat,alpha)

    % Initializations
    nrow = size(imageMat,1);
    ncol = size(imageMat,2);
    lossMat = zeros(nrow,ncol);
    pathMat = zeros(nrow,ncol);
    f_vec = zeros(nrow,1);

    for t=1:nrow
        % Initialize current row loss
        lossMat(t,:) = -alpha*double(imageMat(t,:));
        if(t==1) 
            continue; 
        end
        % For each freq column in rows 2:end
        for f_current=1:ncol
            % Find best previous point
            best_loss_increment = realmax;
            f_best_prev = 0;
            for f_prev=1:ncol
                loss_increment = lossMat(t-1,f_prev)+(1-alpha)*(f_prev-f_current)^2;
                if(loss_increment<best_loss_increment)
                    f_best_prev = f_prev;
                    best_loss_increment = loss_increment;
                end
            end
            % Update loss
            lossMat(t,f_current) = lossMat(t,f_current) + best_loss_increment;
            % Update path
            pathMat(t,f_current) = f_best_prev;
        end
    end

    % Find best end point
    [minLoss,I] = min(lossMat(nrow,:));
    f_vec(nrow) = I;
    % For each row end-1:0
    for t=nrow-1:-1:1
        % Check series[row+1], add to path
        f_vec(t) = pathMat(t+1,f_vec(t+1));
    end

end
