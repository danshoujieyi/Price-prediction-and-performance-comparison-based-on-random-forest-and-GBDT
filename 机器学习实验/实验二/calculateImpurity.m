% 计算熵不纯度
function res = calculateImpurity(examples_)
    P1 = 0;
    P2 = 0;
    [m_,n_] = size(examples_);
    P1 = sum(examples_(:,n_) == '是');
    P2 = sum(examples_(:,n_) == '否');
    P1 = P1 / m_;
    P2 = P2 / m_;
    if P1 == 1 || P1 == 0
        res = 0;
    else
        res = -(P1*log2(P1)+P2*log2(P2));
    end
end
