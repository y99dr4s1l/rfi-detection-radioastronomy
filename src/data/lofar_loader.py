from utils.data import get_lofar_data

##### DEFINE ARGS
class Args:
    def __init__(self):
        # Following master order
        self.model = 'UNET'
        self.anomaly_class = 1
        self.latent_dim = 32
        self.data = 'LOFAR'
        self.data_path = '/kaggle/input/lofar-full-rfi-dataset-pkl/'
        self.seed = 42
        self.patches = True
        self.patch_x = 32
        self.patch_y = 32
        self.patch_stride_x = 32
        self.patch_stride_y = 32
        self.input_shape = (512, 512, 1)
    def update_input_shape(self):
        """Aggiorna input_shape basandosi sulle patch"""
        if self.patches:
            self.input_shape = (self.patch_x, self.patch_y, self.input_shape[-1])

# Uso:
args = Args()
args.update_input_shape()  # Questo aggiorna input_shape quando serve