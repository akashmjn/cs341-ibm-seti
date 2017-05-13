%% For narrow band signals 

aNarrowband = squeeze(narrowbandSignal(2,:,:));
bNarrowband = squeeze(narrowbandSpec(2,:,:));

pxxNarrowband = pwelch(aNarrowband')';

figure();
imagesc(log(bNarrowband));
figure();
imagesc(log(pxxNarrowband));


%% For squiggle signals

aSquiggle = squeeze(squiggleSignal(2,:,:));
bSquiggle = squeeze(squiggleSpec(2,:,:));

pxxSquiggle = pwelch(aSquiggle')';

figure();
imagesc(log(bSquiggle));
figure();
imagesc(log(pxxSquiggle));


%% For squigglesquarepulsednarrowband

aSSPNB = squeeze(squigglesquarepulsednarrowbandSignal(3,:,:));
bSSPNB = squeeze(squigglesquarepulsednarrowbandSpec(3,:,:));

pxxSSPNB = pwelch(aSSPNB')';

figure();
imagesc(log(bSSPNB));
figure();
imagesc(log(pxxSSPNB));
