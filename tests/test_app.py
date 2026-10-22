import pytest
from streamlit.testing.v1 import AppTest
from risklab.config import ROOT


@pytest.mark.parametrize("page",["Market overview","Forecast & breaches","Model backtesting","Stress replay","Inside the network","Sources"])
def test_sections_and_confidence_control(page):
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=40).run()
    assert not app.exception
    app.sidebar.radio[1].set_value(page).run()
    assert not app.exception
    app.sidebar.selectbox[0].set_value(.99).run()
    assert not app.exception
    assert len(app.title) == 1


def test_all_forecast_models_and_size_scaling():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=40).run()
    first=app.metric[0].value
    app.sidebar.number_input[0].set_value(2000000).run()
    assert not app.exception
    assert app.metric[0].value != first
    for model in ["Historical","Gaussian","EWMA"]:
        app.sidebar.selectbox[1].set_value(model).run()
        assert not app.exception


def test_stress_window_and_exposure_control():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=40).run()
    app.sidebar.radio[1].set_value("Stress replay").run()
    next(box for box in app.selectbox if box.label == "Replay window").set_value("Global financial crisis · peak-to-trough window").run()
    assert not app.exception
    app.slider[0].set_value(50).run()
    assert not app.exception


def test_workspaces_share_one_entrypoint_and_return_navigation():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=40).run()
    assert app.sidebar.radio[1].options == ["Market overview","Forecast & breaches","Model backtesting","Stress replay","Inside the network","Sources"]
    app.sidebar.radio[0].set_value("Bitcoin").run()
    assert not app.exception
    assert app.title[0].value=="Bitcoin | Narrative Risk"
    assert app.query_params["view"]==["bitcoin"]
    assert len(app.tabs)==5
    app.sidebar.radio[0].set_value("Equity VaR").run()
    assert not app.exception
    assert len(app.title)==1
    assert app.query_params["view"]==["equity"]


def test_bitcoin_deep_link():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=40)
    app.query_params["view"]="bitcoin"
    app.run()
    assert not app.exception
    assert app.sidebar.radio[0].value=="Bitcoin"
    assert app.title[0].value=="Bitcoin | Narrative Risk"
