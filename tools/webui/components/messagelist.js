import ChatMessage from './chatmessage.js';

const MessageList = (props) => (
    <div className="MessageList">
      {
        props.messages.map(msg => (
          (
            <ChatMessage message={msg.message} party={msg.party}/>
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
    <style jsx>{`
    .MessageList {
      display: grid;
      // grid-template-columns: 100%;
      // overflow-y: auto; 
      // overflow-x:hidden;
    }
    `}</style>
    </div>
  );

export default MessageList;