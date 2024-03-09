encoder_code = """nn.Sequential(
    nn.Linear(self.feature_dim, self.latent_encoder_dim),
    nn.ReLU(),
    nn.Sigmoid()
)"""
decoder_code = """nn.Sequential(
   nn.Linear(self.latent_encoder_dim, self.feature_dim),
    nn.ReLU(),
)""" 



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
degrade_code = """nn.Sequential(nn.Linear(1,2))"""



encoder_code_list = ["""nn.Sequential(
    nn.Linear(self.feature_dim, self.latent_encoder_dim),
    nn.ReLU(),
    nn.Sigmoid()
)""", """nn.Sequential(
    nn.Linear(self.feature_dim, self.latent_encoder_dim),
    nn.ReLU(),
    nn.Sigmoid()
)"""]
decoder_code_list = ["""nn.Sequential(
   nn.Linear(self.latent_encoder_dim, self.feature_dim),
   nn.ReLU(),
)""" ,"""nn.Sequential(
   nn.Linear(self.latent_encoder_dim, self.feature_dim),
   nn.ReLU(),
)"""]