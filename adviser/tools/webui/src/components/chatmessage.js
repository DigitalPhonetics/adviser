import React, {Component} from 'react';

// import './MessageItem.css';

class MessageItem extends Component {


    render() {
        return  <div className={this.props.party}> 
            {this.props.message}
            <p className="party">{this.props.party}</p>
        </div>
    }
}


export default MessageItem;