def make_error_msg(message, detail):
    return {
        'message': message,
        'detail': f'{detail}'
    }


if __name__ == '__main__':
    assert False, make_error_msg('Test', 'This is error test.')