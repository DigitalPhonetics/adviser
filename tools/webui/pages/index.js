import React, {Component} from 'react';
import axios from 'axios';
import ChatWindow from '../components/chatwindow.js';
import TurnInfoList from '../components/turninfolist.js'

class Index extends Component {

  constructor(props) {
    super(props);
    this.state = {
      turnInfo: [{
        index: 0, name: 'test', diff: {}
      }],
      showDebugView: true,
      activeLang: "en"
    };
  }


  receivedTurnInfo = (turn_info) => {
    this.setState({turnInfo: turn_info});
  }


  langSelect_de = () => {
    this.setState({activeLang: "de"})
    this.choose_language("de")
  }

  langSelect_en = () =>  {
    this.setState({activeLang: "en"})
    this.choose_language("en")
  }

  choose_language = (language) => {
    axios.get(`http://localhost:5000/lang`, {params: {lang: language}})
  }

  toggle_debug_view = () => {
    // Toggle debug view
    this.setState({showDebugView: !this.state.showDebugView})
  }

  collapsed_debug_view_style = () => {
    return <style jsx global>{`

        html {
          height: 100vh;
          width: 100vw;
          margin: 0;
          padding: 0;
        }
        body {
          height: 100vh;
          width: 100vw;
          margin: 0px;
          padding: 0px;
          font-family: Sans-serif;
        }
      
        .hsplit {
          height: 100vh;
          display: grid;
          grid-template-columns: 100% 0%;
          // grid-template-rows: 100vh;
        }
        .left {
          background: #e3e3e3;
          grid-column: 1;
          height: calc(100vh - 35px);
          max-height: calc(100vh-35px);
        }
        .right {
          background: white;
          grid-column: 2;
        }
        .languageBar {
          width: 100%;
          padding: 5px;
        }
        .langImg {
          width: 25px;
        }
        .langImgInactive {
          cursor: pointer;
          border-color: transparent;
        }
        .langImgActive {
          border-color: #FF8200;
        }
        `}
        </style>
  }

  expanded_debug_view_style = () => {
    return <style jsx global>{`

        html {
          height: 100vh;
          width: 100vw;
          margin: 0;
          padding: 0;
        }
        body {
          height: 100vh;
          width: 100vw;
          margin: 0px;
          padding: 0px;
          font-family: Sans-serif;
        }
      
        .hsplit {
          height: 100vh;
          display: grid;
          grid-template-columns: 50% 50%;
          // grid-template-rows: 100vh;
        }
        .left {
          background: #e3e3e3;
          grid-column: 1;
          height: calc(100vh - 35px);
          max-height: calc(100vh-35px);
        }
        .right {
          background: white;
          grid-column: 2;
          height: 100vh;
          max-height: 100vh;
        }
        .languageBar {
          width: 100%;
          padding: 5px;
        }
        .langImg {
          width: 25px;
        }
        .langImgInactive {
          cursor: pointer;
          border-color: transparent;
        }
        .langImgActive {
          border-color: #FF8200;
        }
        `}
        </style>
  }


  render = () => (
    <div className='hsplit'>
      <div className="left">
        <div className="languageBar">
          <button onClick={this.toggle_debug_view}>toggle debug view</button>
          {
            this.state.activeLang == "en" ?  
              <img src="/static/img_lang/british_flag.png" className="langImg langImgActive" hspace="5px" border="2"/> :
              <img src="/static/img_lang/british_flag.png" onClick={this.langSelect_en} className="langImg langImgInactive" hspace="5px" border="2"/>
          }
         {
           this.state.activeLang == "de" ?
              <img src="/static/img_lang/german_flag.png"  className="langImg langImgActive" border="2"/> :
              <img src="/static/img_lang/german_flag.png"onClick={this.langSelect_de} className="langImg langImgInactive" border="2"/>
         }
        </div>
        <ChatWindow receivedTurnInfo={this.receivedTurnInfo} />
      </div>
      <div className="right">
        {
          this.state.showDebugView && <TurnInfoList turnInfo={this.state.turnInfo}/>
        }
      </div>
      {
        this.state.showDebugView ? this.expanded_debug_view_style() : this.collapsed_debug_view_style()
      }
    </div>
  );

}
  
export default Index;