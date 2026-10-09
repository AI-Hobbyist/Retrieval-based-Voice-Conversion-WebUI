class APIError(Exception):
    def __init__(self, code, message, status=422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status

    def payload(self, request_id):
        return {"request_id": request_id, "error": {"code": self.code, "message": self.message}}
