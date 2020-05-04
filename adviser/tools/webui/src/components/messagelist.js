import React from "react";
import ChatMessage from './chatmessage.js';

// import '../App.css';

const MessageList = (props) => (
    <div className="MessageList">
      {
        props.messages.map((msg, index) => (
          (
            <ChatMessage key={index} message={msg.message} party={msg.party}/>
          )
        ))
      }
{/*       
      <div className="container">
      <ChatMessage message="test" party="system"/>
      </div>
      <div className="container" >
        <ChatMessage message="test" party="user"/>
      </div> */}
    </div>
  );

export default MessageList;