import React, { Component } from "react";
import socketIOClient from "socket.io-client";
import ChatWindow from './components/chatwindow.js';
import TurnInfoList from './components/turninfolist.js'
import UserStateWindow from './components/userstatewindow';

import flag_en from './static/img_lang/british_flag.png';
import flag_de from './static/img_lang/german_flag.png';

// import './App.css';

class App extends Component {
  constructor() {
    super();
    this.state = {
      turnInfo: [{
        index: 0, name: 'test', diff: {}
      }],
      showDebugView: false,
      activeLang: "en",
      socket: socketIOClient("127.0.0.1:21512/")
      // socket: socketIOClient("127.0.0.1:21512", {
      //     transports: ['websocket']
      //   })
      // response: false,
      // endpoint: "127.0.0.1:21512"
    };
  }
  componentDidMount() {
    // const { endpoint } = this.state;
    // const socket = socketIOClient(endpoint)
    // const socket = socketIOClient(endpoint, {
    //   transports: ['websocket']
    // });
    // socket.on("message", data => { console.log({ data }); socket.send("Test from React") });
    
    // add event listener
    this.state.socket.on("connect", () => console.log("Connected"));
    this.state.socket.on("message", (msg) => console.log(msg))
    // this.state.socket.on("sys_utterance", (msg) => console.log("Got message from dialog system: " + msg));
    // start new dialog
    this.state.socket.emit("start_dialog", true)
  }


  receivedTurnInfo = (turn_info) => {
    this.setState({ turnInfo: turn_info });
  }


  langSelect_de = () => {
    this.setState({ activeLang: "de" })
    this.choose_language("de")
  }

  langSelect_en = () => {
    this.setState({ activeLang: "en" })
    this.choose_language("en")
  }

  choose_language = (language) => {
    this.state.socket.emit('lang', language)
    // axios.get(`http://localhost:5000/lang`, { params: { lang: language } })
  }

  toggle_debug_view = () => {
    // Toggle debug view
    this.setState({ showDebugView: !this.state.showDebugView })
  }

  render = () => (
    <div className='grid-container'>
      <div className="left">
        
        <ChatWindow socket={this.state.socket} receivedTurnInfo={this.receivedTurnInfo} />
      </div>
      <div className="right">
        <UserStateWindow socket={this.state.socket} />
        {/* {
          this.state.showDebugView && <TurnInfoList turnInfo={this.state.turnInfo} />
        } */}
      </div>
    </div>
  );

}

export default App;
