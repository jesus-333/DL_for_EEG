L'idea alla base è avere un modello gerarchico encoder/decoder che trae ispirazione da hvEEGNet e LUNA. Entrambi i paper originali sono in allegato al messagio

Considera che tutti i segnali in input, `x`, saranno campioni EEG, con dimensione `B x 1 x C x T`, con
1) `B` : Batch size
2) `1` : Depth dimension (1 per tutti i segnali di input)
3) `C` : Numero di canali (può cambiare in base al campione)
4) `T` : Numero di segnali temporali (può variare in base al campione)
A prescindere dalla variabilità di `C` e `T` i segnali però dovranno essere tutti campionati a 256Hz e avere lo stesso preprocess (ancora da decidere ma molto probabilmente un qualche filtraggio e una z-score normalization canale per canale, come sul paper di LUNA).

Da LUNA voglio prendere l'idea del channel encoder. Quindi avere un'architettura transformer-like che riesce a mappare un numero arbitrario di canali in una numero fisso `N` canali (quello che loro nel paper chiamano Channel Unification Module).
Questo dovrà essere presente sia in versione encoder (da montatura originale a `N` canali) che decoder (da `N` canali a montatura originale). Voglio sempre tenere qualcosa come le N leaernable queries che il paper di LUNA aveva nel Channel Unification Module.
Quindi dovrò avere qualcosa di questo genere :

```python
x_output = channel_encoder(x_input)

# x_input.shape  = [B x 1 x C x T] 
# x_output.shape = [B x 1 x N x T] 
```

Una volta fatto questo il segnale EEG dovrà essere processato dal resto del modello gerarchico. 
Questo modello dovrà essere composto da moduli con la stessa struttura che trasformano il segnale. Ogni blocco dovrà fare anche un pooling, con fattore 2. L'idea è quella di partire da segnali campionati a 256Hz e "assumere" ogni pooling layer come un downsampling che dimezza la frequenza.
In questo modo più si va in profondità nella rete più le features dovrebero essere relative a basse frequenze e quindi tenere in considerazioni porzioni sempre più ampie di segnale.
So già che il pooling non è un vero downsampling. Sto parlando a un livello più concettuale di dimensione del segnale e features estratte.
L'uscita di oguno di questi moduli inoltre verrà mappata in uno spazio latente, tramite convolutional layer. 
Questi spazi latenti avranno dimensione variabile in quanto il modello dovrà poter ricevere input di lunghezze diverse.
Quindi dovrò avere qualcosa di questo genere :

```python
x_output = hierarchical_block(x_input)
x_latent_space = cnn_map_latent_space(x_output)

# x_input.shape         = [B x 1 x N x T] 
# x_output.shape        = [B x 1 x N x T/2] 
# x_latent_space.shape  = [B x 2 x N x T/2]  
```
Nota che il 2 sul latent space è perchè devo avere una coppia di valori, media e std.

Partendo da segnali di 256Hz sto considerando di avere l'architettura del modello gerarchico composta da 4 moduli :
1) 256Hz --> 128Hz
2) 128Hz --> 64Hz
3) 64Hz --> 32Hz
4) 32Hz --> 16Hz

L'output dell'ultimo blocco dovrà essere a sua volta mappato in uno spazio latente, ma invece di usare una CNN verrà usato un trasformer (o un'altra architettura indipendente dalla dimensione temporale). 
Questo per far si che lo spazio latente più profondo abbia sempre la stessa dimensione.
Quindi dovrò avere qualcosa di questo genere :

```python
x_output = last_hierarchical_block(x_input) 
x_latent_space = cnn_map_latent_space(x_output)

# x_input.shape         = [B x 1 x N x T/8] 
# x_output.shape        = [B x 1 x N x T/16] 
# x_latent_space.shape  = [B x 2 x L]  
```
Dove `L` è la dimensione dello spazio latente, scelta a priori. Questo per far si che lo spazio latente più profondo codifichi veramente informazioni da tutto il segnale.


La mia idea di base, per lo `hierarchical_block` è avere qualcosa di ispirato ad EEGNet. Nota che questo è più pseudocodice che una vera implementazione, giusto per dare un'idea generale.
```python
class hierarchical_block(torch.nn.Module) :
    def __init__(self, config : dict) :
        activation = config["activate"]

        temporal_filter = torch.nn.Conv2d(in_channels = 1, out_channels = config["temporal_filter_out_channels"], kernel_size = ... )
        spatial_filter  = torch.nn.Conv2d(config["temporal_filter_out_channels"] = 1, out_channels = config["temporal_filter_out_channels"], kernel_size = N, groups = config["temporal_filter_out_channels"]  ... )
        separable_conv = torch.nn.Sequential(
            ...
        )

        pooling = torch.nn.AvgPool2d(kernel_size = 2)

    def forward(self, x) :
        x = temporal_filter(x)  # x.shape = [B x temporal_filter_out_channels x C x T]
        x = activation(x)       
        x = spatial_filter(x)   # x.shape = [B x temporal_filter_out_channels x 1 x T]
        x = activation(x)
        x = separable_conv(x)   # x.shape = [B x temporal_filter_out_channels x 1 x T]
        x = pooling(x)          # x.shape = [B x temporal_filter_out_channels x 1 x T/2]
        
        # Reshape to move depth dimension to the channel dimension
        x = x.permute(0, 2, 1, 3) # x.shape = [B x 1 x temporal_filter_out_channels x T/2]

        return x
```

Praticamente è una simil-EEGNet dove però ho solo un pooling layer alla fine e poi faccio uno shift degli assi per riottenere una dimensione di output consistente con il resto dell'architettura.

Per il decoder pensavo di avere una simile architettura ma "flippata". Invertire l'ordine dei layer e usare upsampling layer al posto del pooling. eventualmente anche la transpose convolution al posto della convoluzione standard.

Come training vorrei fare una cosa stile pretraining, usando svariati dataset di molti tipi diversi di EEG. Stavo pensando a tutti i dataset del MOABB più parte del TUEG. 
Come loss function stavo pensando di usare la L1Smooth loss, più efficiente della SoftDTW (eventualmente quest'ultima da testare se le performance del modello sono basse)

Vorrei che tu facessi un analisi approfondità di questa idea, punti di forza e criticità.
Alcuni punti su cui sto riflettendo già io.
1) Usare la depth dimension come canali per i layer successivi. Si tratta della soluzione più facile ma non so quanto sia efficace. Analisi e altre proposte sono ben accette.
2) Il decoder come "flip" dell'encoder, soluzione semplice (già funzionale con hvEEGNet). Però mi sto domandando se ci siano soluzione più eleganti o efficienti.
3) L'assenza di una qualche sorta di attenzione temporale dentro ogni `hierarchical_block`
4) Eventuale modifica anche del penultimo spazio latente per avere un fixed size.
5) Sono ancora molto incerto su come implementare il channel encoder.
6) Modificare il training usando segnali con sampling rate diversi (sempre però multipli o sottomultipli di 256Hz). Questo richiederebbe una modifica dell'architettura come sopra presentata. L'idea di base sarebbe immettere il segnale nel giusto momento della gerarchia più eventualmente una sorta di residual connection come Resnet. Ce l'ho già in testa ma possiamo tenere la discussione per un altro momento.

Ovviamente altre note/criticità/migliorie etc sono benvenute










