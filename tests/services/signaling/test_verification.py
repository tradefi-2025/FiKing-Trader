from src.services.signaling.verification import SignalingVerification


class DummyModel:
    pass


def test_verification_methods_exist():
    verification = SignalingVerification(DummyModel())

    assert verification.verify_inference(None) is None
    assert verification.create_api_request({}) is None
