clear all;
close all;
clc;
 
cd '/Users/daniel/MATLAB/Classifier/Music'
 
musicfiles = dir(pwd);
songname = dir('*wav');
 
songs = [];
sr = 44100; % sampling rate obtained from sampling first few songs
for i = 3:length(musicfiles)
    [Y, FS] = audioread(musicfiles(i).name, [7*sr, 12*sr]);
    signal = Y(:,1) + Y(:,2);
    songs(:,i-2) = signal;
end
 
%%
 
sgrams = zeros(146718,1000);
window = hamming(1024);
noverlap = 256;
for i = 1:length(songs(1,:))
    S = spectrogram(songs(:,i), window, noverlap);
    S = abs(S);
    spsize = size(S);
    S = reshape(S, [spsize(1) * spsize(2), 1]);
    sgrams(:,i) = S;
end
 
%%
 
blues = sgrams(:,1:100);
classical = sgrams(:,101:200);
country = sgrams(:,201:300);
disco = sgrams(:,301:400);
hiphop = sgrams(:,401:500);
jazz = sgrams(:,501:600);
metal = sgrams(:,601:700);
pop = sgrams(:,701:800);
reggae = sgrams(:,801:900);
rock = sgrams(:,901:1000);
 
%%
 
[U1, S1, V1] = svd(blues,'econ');
[U2, S2, V2] = svd(classical,'econ');
[U3, S3, V3] = svd(country,'econ');
[U4, S4, V4] = svd(disco,'econ');
[U5, S5, V5] = svd(hiphop,'econ');
[U6, S6, V6] = svd(jazz,'econ');
[U7, S7, V7] = svd(metal,'econ');
[U8, S8, V8] = svd(pop,'econ');
[U9, S9, V9] = svd(reggae,'econ');
[U10, S10, V10] = svd(rock,'econ');
 
%%
figure(1)
subplot(5,2,1)
plot(diag(S1),'ko')
subplot(5,2,2)
plot(diag(S2),'ko')
subplot(5,2,3)
plot(diag(S3),'ko')
subplot(5,2,4)
plot(diag(S4),'ko')
subplot(5,2,5)
plot(diag(S5),'ko')
subplot(5,2,6)
plot(diag(S6),'ko')
subplot(5,2,7)
plot(diag(S7),'ko')
subplot(5,2,8)
plot(diag(S8),'ko')
subplot(5,2,9)
plot(diag(S9),'ko')
subplot(5,2,10)
plot(diag(S10),'ko')
 
%%
mode_low = 1;
mode_high = 10;
 
%%
 
figure(2)
plot3(V1(:,mode_low),V1(:,mode_low+1),V1(:,mode_low+2),'o')
hold on
plot3(V2(:,mode_low),V2(:,mode_low+1),V2(:,mode_low+2),'o')
hold on
plot3(V3(:,mode_low),V3(:,mode_low+1),V3(:,mode_low+2),'o')
hold on
plot3(V4(:,mode_low),V4(:,mode_low+1),V4(:,mode_low+2),'o')
hold on
plot3(V5(:,mode_low),V5(:,mode_low+1),V5(:,mode_low+2),'o')
hold on
plot3(V6(:,mode_low),V6(:,mode_low+1),V6(:,mode_low+2),'o')
hold on
plot3(V7(:,mode_low),V7(:,mode_low+1),V7(:,mode_low+2),'o')
hold on
plot3(V8(:,mode_low),V8(:,mode_low+1),V8(:,mode_low+2),'o')
hold on
plot3(V9(:,mode_low),V9(:,mode_low+1),V9(:,mode_low+2),'o')
hold on
plot3(V10(:,mode_low),V10(:,mode_low+1),V10(:,mode_low+2),'o')
legend('blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock');
 
%% Training with Cross-Validation
 
totalscores1 = [];
totalscores2 = [];
totalscores3 = [];
 
 
c_v_iter = 5;
 
for i = 1:c_v_iter
 
q1 = randperm(100);
q2 = randperm(100);
q3 = randperm(100);
q4 = randperm(100);
q5 = randperm(100);
q6 = randperm(100);
q7 = randperm(100);
q8 = randperm(100);
q9 = randperm(100);
q10 = randperm(100);
 
xblues = V1(:,mode_low:mode_high);
xclassical = V2(:,mode_low:mode_high);
xcountry = V3(:,mode_low:mode_high);
xdisco = V4(:,mode_low:mode_high);
xhiphop = V5(:,mode_low:mode_high);
xjazz = V6(:,mode_low:mode_high);
xmetal = V7(:,mode_low:mode_high);
xpop = V8(:,mode_low:mode_high);
xreggae = V9(:,mode_low:mode_high);
xrock = V10(:,mode_low:mode_high);
 
xtrain = [xblues(q1(1:70),:)
    xclassical(q2(1:70),:)
    xcountry(q3(1:70),:)
    xdisco(q4(1:70),:)
    xhiphop(q5(1:70),:)
    xjazz(q6(1:70),:)
    xmetal(q7(1:70),:)
    xpop(q8(1:70),:)
    xreggae(q9(1:70),:)
    xrock(q10(1:70),:)];
 
xtest = [xblues(q1(71:100),:)
    xclassical(q2(71:100),:)
    xcountry(q3(71:100),:)
    xdisco(q4(71:100),:)
    xhiphop(q5(71:100),:)
    xjazz(q6(71:100),:)
    xmetal(q7(71:100),:)
    xpop(q8(71:100),:)
    xreggae(q9(71:100),:)
    xrock(q10(71:100),:)];
 
%%
 
ctrain = [ones(70,1)
    2*ones(70,1)
    3*ones(70,1)
    4*ones(70,1)
    5*ones(70,1)
    6*ones(70,1)
    7*ones(70,1)
    8*ones(70,1)
    9*ones(70,1)
    10*ones(70,1)];
 
%%
 
ctrain3 = [ones(70,3)
    2*ones(70,3)
    3*ones(70,3)
    4*ones(70,3)
    5*ones(70,3)
    6*ones(70,3)
    7*ones(70,3)
    8*ones(70,3)
    9*ones(70,3)
    10*ones(70,3)];
 
%% Naive Bayes
 
nb = fitcnb(xtrain,ctrain);
pre1 = nb.predict(xtest);
 
bar(pre1)
score = 0;
scores1 = zeros(1,10);
for i = 1:length(pre1)
    if pre1(i) == (floor(i/30) + 1)
        score = score + 1;
    end
    if i~= 1 && (mod(i,30) == 1)
        scores1(floor(i/30)) = score / 30.0;
        score = 0;
    end
    if i == length(pre1)
        scores1(floor(i/30)) = score / 30.0;
    end
end
 
%% Linear Discriminant Analysis
 
pre2 = classify(xtest,xtrain,ctrain);
bar(pre2)
score = 0;
scores2 = zeros(1,10);
for i = 1:length(pre2)
    if pre2(i) == (floor(i/30) + 1)
        score = score + 1;
    end
    if i~= 1 && (mod(i,30) == 1)
        scores2(floor(i/30)) = score / 30.0;
        score = 0;
    end
    if i == length(pre2)
        scores2(floor(i/30)) = score / 30.0;
    end
end
 
%% k-nearest neighbors
 
[IDX, D] = knnsearch(xtrain,xtest);
pre3=[];
for i=1:length(xtest(:,1))
    pre3(i) = ctrain(IDX(i));
end
bar(pre3);
scores3 = zeros(1,10);
for i = 1:length(pre3)
    if pre3(i) == (floor(i/30) + 1)
        score = score + 1;
    end
    if i~= 1 && (mod(i,30) == 1)
        scores3(floor(i/30)) = score / 30.0;
        score = 0;
    end
    if i == length(pre3)
        scores3(floor(i/30)) = score / 30.0;
    end
end
 
totalscores1 = [totalscores1; scores1];
totalscores2 = [totalscores2; scores2];
totalscores3 = [totalscores3; scores3];
 
end
 
averagescores1 = [];
averagescores2 = [];
averagescores3 = [];
for i = 1:length(scores1)
    averagescores1(:,i) = sum(totalscores1(:,i)) / c_v_iter;
    averagescores2(:,i) = sum(totalscores2(:,i)) / c_v_iter;
    averagescores3(:,i) = sum(totalscores3(:,i)) / c_v_iter;
end
 
%%
 
genrenames = {'blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'};
 
tablem1 = [totalscores1; averagescores1].';
tablem2 = [totalscores2; averagescores2].';
tablem3 = [totalscores3; averagescores3].';
 
table1 = table(tablem1(:,1), tablem1(:,2), tablem1(:,3), tablem1(:,4), tablem1(:,5), tablem1(:,6));
table2 = table(tablem2(:,1), tablem2(:,2), tablem2(:,3), tablem2(:,4), tablem2(:,5), tablem2(:,6));
table3 = table(tablem3(:,1), tablem3(:,2), tablem3(:,3), tablem3(:,4), tablem3(:,5), tablem3(:,6));
 
table1.Properties.VariableNames = {'Trial1' 'Trial2' 'Trial3' 'Trial4' 'Trial5' 'Average'};
table2.Properties.VariableNames = {'Trial1' 'Trial2' 'Trial3' 'Trial4' 'Trial5' 'Average'};
table3.Properties.VariableNames = {'Trial1' 'Trial2' 'Trial3' 'Trial4' 'Trial5' 'Average'};
table1.Properties.RowNames = genrenames;
table2.Properties.RowNames = genrenames;
table3.Properties.RowNames = genrenames;
