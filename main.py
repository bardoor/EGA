import telebot
from telebot import types

bot = telebot.TeleBot('6883397009:AAE_zS_E1I8iTMuwu1AkHKXQMywJV-mZeQ0')


@bot.message_handler(commands=['start'])
def newChat(message):
    bot.send_message(message.chat.id,
                     f'Здравствуйте, <b>{message.from_user.first_name}</b>.\nС каким <b><u>социальным вопросом</u></b> вам необходимо помочь?',parse_mode='HTML')


@bot.message_handler(commands=['help'])
def helpCommand(message):
    bot.send_message(message.chat.id, f'<b>Dvora</b> может помочь вам с решением разного рода социальных вопросов.\n\n'
                                      f'<b>Доступные команды:</b>\n'
                                      f'/start - начало работы с ботом\n'
                                      f'/help - помощь по боту\n'
                                      f'/spheres - сферы, с которыми работает бот\n\n'
                                      f'<b>ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ DVORA:</b>\n'
                                      f'1) Введите свой социальный вопрос, с которым вам нужно помочь.\n'
                                      f'2) Dvora бот самостоятельно определит сферу и тему вашего обращения.\n'
                                      f'3) Далее, Drova отправит запрос в отделение, ответственное за решение данного вопроса.', parse_mode='HTML')


@bot.message_handler(commands=['spheres'])
def spheresCommand(message):
    bot.send_message(message.chat.id, f'<b>Сферы, в которых бот может вам помочь:</b>\n\n'
                                      f'<em>Безопасность</em>\n'
                                      f'<em>Благоустройство</em>\n'
                                      f'<em>Газ и топливо</em>\n'
                                      f'<em>Благоустройство</em>\n'
                                      f'<em>Государственная собственность</em>\n'
                                      f'<em>Дороги</em>\n'
                                      f'<em>ЖКХ</em>\n'
                                      f'<em>Здравоохранение/Медицина</em>\n'
                                      f'<em>Здравоохранение/Медицина</em>\n'
                                      f'<em>Коронавирус</em>\n'
                                      f'<em>Культура</em>\n'
                                      f'<em>МФЦ/Мои документы</em>\n'
                                      f'<em>Мобилизация</em>\n'
                                      f'<em>Мусор/Свалки/ТКО</em>\n'
                                      f'<em>Образование</em>\n'
                                      f'<em>Общественный транспорт</em>\n'
                                      f'<em>Памятники и объекты культурного наследия</em>\n'
                                      f'<em>Погребение и похоронное дело</em>\n'
                                      f'<em>Роспотребнадзор</em>\n'
                                      f'<em>Связь и телевидение</em>\n'
                                      f'<em>Социальное обслуживание и защита</em>\n'
                                      f'<em>Спецпроекты</em>\n'
                                      f'<em>Строительство и архитектура</em>\n'
                                      f'<em>Торговля</em>\n'
                                      f'<em>Физическая культура и спорт</em>\n'
                                      f'<em>Экология</em>\n'
                                      f'<em>Экономика и бизнес</em>\n'
                                      f'<em>Электроснабжение</em>\n', parse_mode='HTML')


@bot.message_handler()
def answerHello(message):
        bot.reply_to(message, f'Привет, {message.from_user.first_name}')



bot.infinity_polling()
