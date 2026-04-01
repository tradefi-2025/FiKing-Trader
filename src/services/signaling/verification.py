class SignalingVerification:

    def __init__(self, model):
        self.model = model

    def verify(self, input_data):
        return{
            "is_valid": True
        }

    def create_api_request(self, payload):
        """Create API request payload"""
        pass
