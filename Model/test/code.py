# encoder_code = """nn.Sequential(
#     nn.Linear(self.feature_dim, self.latent_encoder_dim),
#     nn.ReLU(),
#     nn.Sigmoid()
# )"""
# decoder_code = """nn.Sequential(
#    nn.Linear(self.latent_encoder_dim, self.feature_dim),
#     nn.ReLU(),
# )""" 



# encoder_code_list = ["""nn.Sequential(
#     nn.Linear(self.feature_dim, self.latent_encoder_dim),
#     nn.ReLU(),
#     nn.Sigmoid()
# )""", """nn.Sequential(
#     nn.Linear(self.feature_dim, self.latent_encoder_dim),
#     nn.ReLU(),
#     nn.Sigmoid()
# )""","""nn.Sequential(
#     nn.Linear(self.feature_dim, self.latent_encoder_dim),
#     nn.ReLU(),
#     nn.Sigmoid()
# )""", """nn.Sequential(
#     nn.Linear(self.feature_dim, self.latent_encoder_dim),
#     nn.ReLU(),
#     nn.Sigmoid()
# )""","""nn.Sequential(
#     nn.Linear(self.feature_dim, self.latent_encoder_dim),
#     nn.ReLU(),
#     nn.Sigmoid()
# )""", """nn.Sequential(
#     nn.Linear(self.feature_dim, self.latent_encoder_dim),
#     nn.ReLU(),
#     nn.Sigmoid()
# )"""]
# decoder_code_list = ["""nn.Sequential(
#    nn.Linear(self.latent_encoder_dim, self.feature_dim),
#    nn.ReLU(),
# )""" ,"""nn.Sequential(
#    nn.Linear(self.latent_encoder_dim, self.feature_dim),
#    nn.ReLU(),
# )""","""nn.Sequential(
#    nn.Linear(self.latent_encoder_dim, self.feature_dim),
#    nn.ReLU(),
# )""" ,"""nn.Sequential(
#    nn.Linear(self.latent_encoder_dim, self.feature_dim),
#    nn.ReLU(),
# )""","""nn.Sequential(
#    nn.Linear(self.latent_encoder_dim, self.feature_dim),
#    nn.ReLU(),
# )""" ,"""nn.Sequential(
#    nn.Linear(self.latent_encoder_dim, self.feature_dim),
#    nn.ReLU(),
# )"""]




encoder_code_list = ["""nn.Sequential(
    nn.Linear(self.feature_dim,1024),
    nn.ReLU(), 
    nn.Linear(1024, 1024),
    nn.ReLU(),            
    nn.Linear(1024, self.latent_encoder_dim)
)"""] * 2
decoder_code_list = ["""nn.Sequential(
    nn.Linear(self.latent_encoder_dim, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, self.feature_dim),
    nn.Sigmoid()
)"""] * 2
degrade_code = ["""nn.Sequential(
    nn.Linear(self.h_dim, 2 * self.latent_encoder_dim),
    nn.ReLU(),
    nn.Linear(2 * self.latent_encoder_dim, self.h_dim),
    nn.ReLU()
)"""] * 2