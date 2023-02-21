Collate2 Funkcia funguje pre obe maskrcnn a fpn50 
ale problematicke je ze to berie len 3 channels
mozno sa vratim spat ku tomu co som chcel originalne robit
Intensity + depth + nieco  aby to bolo 3 kanalove
ak mam zvysovat kanaly je to pain - vobec nie jednoduche

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn2) seems to work fine akurat naplni gpu which is fucking wierd
for bigger batch sizes bigger gpu is sorely needed 
co je zaujimave je ze to padne az na 2 batchi 
solved by using checkpoininh toto treba isto spomenut

current problem - gradscaler pada na nejakom inf checku 
dunno what thats about bez scaleru to ide