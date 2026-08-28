"""Minh hoạ FUNCTION CALLING thuần với Google Gemini SDK.

Tool `get_weather` được định nghĩa schema thủ công VÀ thực thi ngay trong
chính file app này. Model chỉ QUYẾT ĐỊNH gọi tool nào; app mới là nơi chạy.

Cách chạy:
    pip install -r ../requirements.txt
    export GEMINI_API_KEY=...
    python weather_function_calling.py
"""

import json

import httpx
from google import genai
from google.genai import types

client = genai.Client()

MODEL = "gemini-3.6-flash"

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thời tiết thân thiện, trả lời bằng tiếng Việt tự nhiên. "
    "Dùng emoji phù hợp (🌧️ 🌤️ 💨 💧). "
    "Tóm tắt ngắn gọn, dễ hiểu, và đưa ra lời khuyên thực tế "
    "(ví dụ: mang ô, mặc áo mỏng, ...)."
)

# 1. App tự định nghĩa schema của tool
get_weather_declaration = types.FunctionDeclaration(
    name="get_weather",
    description="Lấy thời tiết hiện tại của một thành phố",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(
                type=types.Type.STRING, description="Tên thành phố"
            )
        },
        required=["city"],
    ),
)

# 1b. Tool thứ hai — dự báo nhiều ngày, gọi API thời tiết THẬT (không mock)
get_forecast_declaration = types.FunctionDeclaration(
    name="get_forecast",
    description="Lấy dự báo thời tiết nhiều ngày tới của một thành phố",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(
                type=types.Type.STRING, description="Tên thành phố"
            ),
            "days": types.Schema(
                type=types.Type.INTEGER,
                description="Số ngày muốn dự báo (1-7, mặc định 3)",
            ),
        },
        required=["city"],
    ),
)

TOOLS = [
    types.Tool(
        function_declarations=[get_weather_declaration, get_forecast_declaration]
    )
]


# 2. App tự thực thi tool (trong thực tế sẽ gọi API thời tiết thật)
def get_weather(city: str) -> str:
    """Trả về thời tiết (mock) của *city*. Dùng làm tool cho model."""
    mock_data = {
        "Hà Nội": {
            "nhiệt_độ": "29°C",
            "thời_tiết": "trời mưa nhẹ",
            "độ_ẩm": "82%",
            "gió": {"hướng": "Đông Nam", "tốc_độ": "12 km/h"},
        },
        "Hồ Chí Minh": {
            "nhiệt_độ": "33°C",
            "thời_tiết": "mưa rào",
            "độ_ẩm": "75%",
            "gió": {"hướng": "Tây Nam", "tốc_độ": "15 km/h"},
        },
        "Đà Nẵng": {
            "nhiệt_độ": "30°C",
            "thời_tiết": "nhiều mây",
            "độ_ẩm": "78%",
            "gió": {"hướng": "Đông", "tốc_độ": "10 km/h"},
        },
    }
    default = {"nhiệt_độ": "28°C", "thời_tiết": "không có dữ liệu chi tiết"}
    return json.dumps({"city": city, **mock_data.get(city, default)}, ensure_ascii=False)


# WMO weather code -> mô tả tiếng Việt (chuẩn dùng bởi Open-Meteo)
WMO_CODE_VI = {
    0: "trời quang", 1: "quang, ít mây", 2: "có mây rải rác", 3: "nhiều mây",
    45: "sương mù", 48: "sương mù đóng băng",
    51: "mưa phùn nhẹ", 53: "mưa phùn vừa", 55: "mưa phùn dày",
    61: "mưa nhỏ", 63: "mưa vừa", 65: "mưa to",
    71: "tuyết nhẹ", 73: "tuyết vừa", 75: "tuyết dày",
    80: "mưa rào nhẹ", 81: "mưa rào vừa", 82: "mưa rào dữ dội",
    95: "dông", 96: "dông kèm mưa đá nhẹ", 99: "dông kèm mưa đá to",
}


def get_forecast(city: str, days: int = 3) -> str:
    """Gọi API thời tiết THẬT (Open-Meteo, không cần API key) để lấy dự báo
    *days* ngày tới cho *city*. Không dùng mock data.
    """
    days = max(1, min(days, 7))

    # Bước 1: geocode tên thành phố -> toạ độ (Open-Meteo Geocoding API)
    geo_resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "vi", "format": "json"},
        timeout=10.0,
    )
    geo_resp.raise_for_status()
    geo_results = geo_resp.json().get("results")
    if not geo_results:
        return json.dumps({"city": city, "error": "Không tìm thấy thành phố"}, ensure_ascii=False)

    location = geo_results[0]
    lat, lon = location["latitude"], location["longitude"]

    # Bước 2: lấy dự báo theo ngày (Open-Meteo Forecast API)
    forecast_resp = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=10.0,
    )
    forecast_resp.raise_for_status()
    daily = forecast_resp.json()["daily"]

    forecast_days = [
        {
            "ngày": daily["time"][i],
            "nhiệt_độ_thấp": f"{daily['temperature_2m_min'][i]}°C",
            "nhiệt_độ_cao": f"{daily['temperature_2m_max'][i]}°C",
            "lượng_mưa": f"{daily['precipitation_sum'][i]}mm",
            "thời_tiết": WMO_CODE_VI.get(daily["weathercode"][i], "không xác định"),
        }
        for i in range(len(daily["time"]))
    ]

    return json.dumps(
        {"city": location.get("name", city), "dự_báo": forecast_days},
        ensure_ascii=False,
    )


TOOL_FUNCTIONS = {"get_weather": get_weather, "get_forecast": get_forecast}


def run(prompt: str) -> str:
    """Gửi *prompt* tới Gemini, tự động xử lý function calling và trả về câu trả lời cuối."""
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    # 3. Gọi model — model quyết định có gọi tool hay không
    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    # 4. Vòng lặp: nếu model yêu cầu tool, app TỰ THỰC THI rồi đưa kết quả trả lại
    while resp.function_calls:
        # Thêm phản hồi của model vào lịch sử hội thoại
        contents.append(resp.candidates[0].content)

        function_responses = []
        for fc in resp.function_calls:
            print(f"  [model yêu cầu] {fc.name}({fc.args})")
            result = TOOL_FUNCTIONS[fc.name](**fc.args)  # <-- app chạy, không phải model
            print(f"  [app thực thi]  -> {result}")
            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": result}
                )
            )

        # Gửi kết quả tool trả về cho model
        contents.append(types.Content(role="user", parts=function_responses))
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    # 5. Model tổng hợp câu trả lời cuối
    return resp.text


if __name__ == "__main__":
    question = "Thời tiết Hà Nội hôm nay thế nào, và dự báo 3 ngày tới ở Đà Nẵng ra sao?"
    print(f"User: {question}\n")
    print("Trả lời:", run(question))
